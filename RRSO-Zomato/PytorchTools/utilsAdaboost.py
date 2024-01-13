# -*- coding: utf-8 -*-

"""
@author: Alexandre

Note: here are the functions used in the Adaboost 

"""
# Utilities for Pytorch
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics import AUROC

# Set CUDA:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# Constants:
num_class = 3  # define the number of classes 

#==============================================================================
# Helper function for training our model for one epoch:
def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples, w_i, batch_size):
  model = model.train()
  
  N=batch_size # batch size 128
  losses = []
  correct_predictions = 0  
  
  for i,d in enumerate(tqdm(data_loader)):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(input_ids=input_ids,
                  attention_mask=attention_mask)
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)
    
    sample_weight = w_i[i*N : (i+1)*N]
    sample_weight = torch.from_numpy(sample_weight).to(device)
    
    # print (sample_weight, loss)
    loss = (loss * sample_weight / sample_weight.sum()).sum()
    
    # print ('/n loss', loss.item())
    
    # loss = w_i_batch*loss_per_sample
    
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())
    
    loss.mean().backward()
    # loss.backward()
    
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    # scheduler.step() #!!!
    optimizer.zero_grad()
    probs = F.softmax(outputs, dim=1)
    
    auroc = AUROC(num_classes=num_class, task="multiclass")
    auc = auroc(probs, targets)
  return correct_predictions.double() / n_examples, np.mean(losses), auc

#==============================================================================
# Evaluate the model on a given data loader:
def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(input_ids=input_ids,
                      attention_mask=attention_mask)
      _, preds = torch.max(outputs, dim=1)
      loss_per_sample = loss_fn(outputs, targets)
      loss = torch.mean(loss_per_sample)
      
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
      probs = F.softmax(outputs, dim=1)
      
      auroc = AUROC(num_classes=num_class)
      auc = auroc(probs, targets)
  return correct_predictions.double() / n_examples, np.mean(losses), auc

#==============================================================================
# Function to get the predictions from our model:
def get_predictions(model, data_loader):
  model = model.eval()
  
  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      texts = d["review_text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(input_ids=input_ids,
                      attention_mask=attention_mask)
      _, preds = torch.max(outputs, dim=1)
      #Get the predicted probs from our trained model, applying the softmax 
      probs = F.softmax(outputs, dim=1)

      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values

#==============================================================================
# Helper function to create a couple of data loaders:
def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = ReviewDataset(reviews=df.text.to_numpy(),
                    targets=df.rating.to_numpy(),
                    tokenizer=tokenizer,
                    max_len=max_len)

  return DataLoader( ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0) # should be 0 on Windows

#==============================================================================
# Building blocks required to create a PyTorch dataset:
class ReviewDataset:
  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.reviews)
  
  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]
    
    encoding = self.tokenizer.encode_plus(review,
                                          add_special_tokens=True,
                                          max_length=self.max_len,
                                          return_token_type_ids=False,
                                          pad_to_max_length=True,
                                          return_attention_mask=True,
                                          return_tensors='pt')
    return {'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)}


#================================ END FFUNCTIONS ==============================
