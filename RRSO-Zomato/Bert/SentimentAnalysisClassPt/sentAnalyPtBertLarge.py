# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 21:43:35 2023

@author: Alexandre
"""

# Import libraries:
import torch
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torchmetrics import AUROC
from pycm import ConfusionMatrix
from transformers import logging
from collections import defaultdict
from time import time, strftime, gmtime
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
# Libraries created localy:
from PytorchTools import utils as ut
from EarlyStop import earlyStop as es
from PytorchTools.getMetrics import show_train_history
from PytorchTools.mySaveMetrics import selectMetrics, save, create_dir

# To ignore deprecated warnings:
warnings.filterwarnings("ignore")
# BERT warning --> this warning means that during your training, you're not using the pooler in order to compute the loss
logging.set_verbosity_error()

# set CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Constants:
freeze_bert = False # freeze BERT model to train only the classifier

# Save name for metrics and checkpoint:
save_name = 'SentAnalysisPtBertLarge' # change the name to avoid overwrite
save_metrics = True # save the best model parameters
save_model = True # save the model

# Hyperparameters:  #!!! influenciam o treino..................................................
max_len = 100
batch_size = 64
epochs = 20
patience = 10  # need to increase from 10 to --->
delta = 0.0003   # need to decrease to have 4 decimals
Dropout = 0.2
l_r = 2e-5

# Set Path for save checkpoint:
path = './NewBert/SentimentAnalysisClassPt/'
# set path for save the model:
path_model = path + 'Model_' + save_name + '/'

# Language
lang = 'Pt'
num_class = 3
data_path = './Data/FinalData' + lang + '/'
class_names = ['Negative', 'Neutral', 'Positive']

# Set the name to import the model:
model_name = 'neuralmind/bert-large-portuguese-cased'

#%% Load Dataset:
# Load data 80% of dataset for training
df_train = pd.read_csv(data_path + 'trainData_' + str(num_class) + 'Class_'+lang +'.csv', index_col=0)
# Load 70% of the 20% remaining of dataset for valifation
df_val = pd.read_csv(data_path + 'valData_' + str(num_class) + 'Class_'+lang +'.csv', index_col=0)
# Load 30% of the 20% remaining of dataset for testing
df_test = pd.read_csv(data_path + 'testData_' + str(num_class) + 'Class_'+lang +'.csv', index_col=0)

# Load Tokenizer:
tokenizer = BertTokenizer.from_pretrained(model_name) # BertTokenizer 

# Creating Data Loader:
train_data_loader = ut.create_data_loader(df_train, tokenizer, max_len, batch_size)
val_data_loader = ut.create_data_loader(df_val, tokenizer, max_len, batch_size)
test_data_loader = ut.create_data_loader(df_test, tokenizer, max_len, batch_size)

#%% Classifier using BERT:
# Create a classifier that uses the BERT model
class SentimentClassifier(nn.Module):
  def __init__(self, n_classes, model_name):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(model_name, return_dict=False)
    # Freeze BERT parameters:
    if freeze_bert:
        # freeze all the parameters excluding classifier
        for param in self.bert.named_parameters():
            # param = (nome_da_layer, parametros)
            param[1].requires_grad = False
    self.drop = nn.Dropout(Dropout) # layer for regulation
    #The last_hidden_state is a sequence of hidden states of the last layer of the model
    self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask)
    output = self.drop(pooled_output) 
    output = self.linear(output)  
    return output

#%% Full model to GPU and compute weights:
model = SentimentClassifier(len(class_names),model_name)
model = model.to(device)

data = next(iter(train_data_loader))
input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

# Train parameters optimizer:
# optimizer = torch.optim.AdamW(model.parameters(), lr=l_r) 
optimizer = torch.optim.Adagrad(model.parameters(), lr=l_r) # this is the correct

# Scheduler function:
total_steps = len(train_data_loader) * epochs
warmup_steps = total_steps*0.1 # 10% de total_steps
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, 
                                            num_training_steps=total_steps)

# Compute the Class Weights:
class_weights = compute_class_weight(class_weight = "balanced",
                                     classes = np.unique(df_train['rating']),
                                     y = df_train['rating'])
class_wts_dic = dict(zip(np.unique(df_train['rating']), class_weights))
class_wts = [class_wts_dic[0]]
for index in range(1,np.count_nonzero(np.unique(df_train['rating']))+1):
    class_wts.append(class_wts_dic[index])

# Loss function:
loss_fn = nn.CrossEntropyLoss(weight = torch.tensor(class_wts,dtype=torch.float)).to(device)

#%% Early stopping:
early_stopping = es.EarlyStopping(verbose=True, patience=patience, optimizer=optimizer, 
                                  path=path, path_name=save_name, delta=delta)

history = defaultdict(list)
best_accuracy = 0
t = time()
print("\nTraining started...")

# log for  final visualization
log = ''
log += " Log of classifier: " + save_name + "\n"
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    # Train function:
    train_acc, train_loss, train_auc = ut.train_epoch(model,train_data_loader,loss_fn,optimizer, 
                                                      device,scheduler,len(df_train))
    print(f'Train:  loss {train_loss:.3f}   accuracy {train_acc:.3f}')
    
    # Validation function:
    val_acc, val_loss, val_auc = ut.eval_model(model,val_data_loader,loss_fn, 
                                               device,len(df_val))
    
    print(f'Val:    loss {val_loss:.3f}   accuracy {val_acc:.3f}')
    print()
    
    # Save metrics histoty:
    history['train_acc'].append(train_acc.item())
    history['train_loss'].append(train_loss)
    history['train_auc'].append(train_auc.item())
    history['val_acc'].append(val_acc.item())
    history['val_loss'].append(val_loss)
    history['val_auc'].append(val_auc.item())
    
    # follow the accuracy metric:
    if val_acc > best_accuracy:
        # torch.save(model.state_dict(), './Models/best_model_state.bin')
        best_accuracy = val_acc
        print(f'\n Best Accuracy: {best_accuracy.item():.3f} ')
    
    # early stopping
    early_stopping(val_auc.item(), model)
    if early_stopping.early_stop:
       print("\n Stop at epoch:", epoch+1, "\n")
       break
 
trainTime = time()-t
print(f"Training finished...(Elapsed time: {strftime('%H:%M:%S', gmtime(trainTime))})")

# Plot training Losses:
loss_hist = (history['train_loss'].copy(), history['val_loss'].copy())
show_train_history(loss_hist, 'loss', early_stopping.best_epoch, save_name)
# Plot training Accuracy
acc_hist = (history['train_acc'].copy(), history['val_acc'].copy())
show_train_history(acc_hist, 'acc', early_stopping.best_epoch, save_name)
# Plot training AUC
auc_hist = (history['train_auc'].copy(), history['val_auc'].copy())
show_train_history(auc_hist, 'auc', early_stopping.best_epoch, save_name)

# Load best model parameters:
load_path= path + 'checkpoint_' + save_name +'.pth'
checkpoint = torch.load(load_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Save the best model
if save_model:
    model_path = create_dir(path[:-1], f"Model_{save_name}")
    torch.save({'epoch': epoch,
           'model_state_dict': model.state_dict(),
           'acc': val_acc}, path_model + 'model' + save_name + '.pth')
    print("\nFull Model Save...")

log += f" Training time: {strftime('%H:%M:%S', gmtime(trainTime))}\n"
log += f" Best epoch: {early_stopping.best_epoch}\n"

#%% Predictions:
print("\nTesting model...\n")
y_review_texts, y_pred, y_pred_probs, y_test = ut.get_predictions(model,
                                                                  test_data_loader)

# Classification report:
classTest = classification_report(y_test, y_pred, target_names=class_names)
# print(classTest) # visualização teste

cmTest = ConfusionMatrix(actual_vector= y_test.tolist(), predict_vector = y_pred.tolist())
# print (f'\n === Confusion Matrix ===\n  \n {cmTest}')  # visualização teste

# Confusion Matrix:
cmTest.plot(cmap=plt.cm.Blues, number_label=True,
            normalized=True,    # Using normalized because dataset in imbalanced
            plot_lib="seaborn",
            title="Confusion Matrix")
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.ylabel('True Sentiment')
plt.xlabel('Predicted Sentiment')
plt.xticks(np.arange(3)+0.5, class_names)
plt.yticks(np.arange(3)+0.5, class_names)
plt.show()

# compute AUC_ROC:
auroc = AUROC(num_classes=num_class)
ovr_AUC = auroc(y_pred_probs, y_test).item()

#%% Store metrics:
# Names of metrics to store from PyCM
LabelsOverall = [k for k,v in cmTest.overall_stat.items() if isinstance(v,(float,int)) ]
LabelsClass   = [k for k,v in cmTest.class_stat.items() if isinstance(v[0],(float,int))]

# Extract metrics from cm object
rawOverall = selectMetrics(cmTest.overall_stat,LabelsOverall) + [ovr_AUC]
rawClass   = np.asarray([list(d.values()) for d in selectMetrics(cmTest.class_stat,LabelsClass)])

# Convert into dictionary
mOverall = {k:v for k,v in zip( LabelsOverall, rawOverall )}
mClass = {k:{j:v for j,v in enumerate(varray)} for k,varray in zip( LabelsClass, rawClass )}

log += f" Accuracy: {rawOverall[0]:.3f}\n"
log += f" AUC value: {ovr_AUC:.3f}\n"
log += f" F1 Macro: {mOverall['F1 Macro']:.3f}\n"
log += f"\n Classification report \n{classTest}\n"

# Print Log:
print(log,sep='\n')
with open(f'{path_model}log.txt','a') as log_file:
    log_file.write(log)

# Save the model
if save_metrics:
    save(save_name, path=path,
        trainBestEpoch = [early_stopping.best_epoch],
        trainTime = [trainTime],
        trainLOSS = loss_hist,
        trainACC = acc_hist,
        trainAUC = auc_hist,
        pycmFinalAUC = ovr_AUC,
        pycmOverall = mOverall,
        pycmClass = mClass,
        pycmCM = cmTest.to_array(normalized=True)) # confussion matrix

#================================= END CODE ===================================
