"""
@author: Alexandre

Note: This script is for compute the sentiment analysis with cross-validation 
using StratifiedKFold.

--> este código já correu e salvou os modelos (checkpoint fold 0 e fold 1), métricas e full model.

"""
# Import libraries:
import torch
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torchmetrics import AUROC
from transformers import logging
from time import time, strftime, gmtime
from collections import defaultdict
from pycm import ConfusionMatrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup

# Libraries created localy:
from PytorchTools import utils as ut
from PytorchTools.getMetrics import show_train_history
from PytorchTools.mySaveMetrics import selectMetrics, save, create_dir
from EarlyStop import earlyStop as es

# To ignore deprecated warnings:
warnings.filterwarnings("ignore")
# BERT warning --> this warning means that during your training, you're not using the pooler in order to compute the loss
logging.set_verbosity_error()

# Set CUDA:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Constants:
freeze_bert = False # freeze BERT model to train only the classifier

# Save name for metrics and checkpoint:
save_name = 'SentAnalysisPtCrossValidation' # change the name to avoid overwrite
save_metrics = True # save the best model parameters
save_model = True # save the model

# Hyperparameters:
max_len = 100
batch_size = 128
epochs = 20
patience = 10  # need to increase from 10 to --->
delta = 0.0003   # need to decrease to have 4 decimals
Dropout = 0.2
l_r = 2e-5

# Set Path for save checkpoint:
path = './NewBert/SentimentAnalysisClassPtCrossValidation/'
# set path for save the model:
path_model = path + 'Model_' + save_name + '/'

# Language
lang = 'Pt'
num_class = 3
data_path = './Data/FinalData' + lang + '/'
class_names = ['Negative', 'Neutral', 'Positive']

# Set the name to import the model:
model_name = 'neuralmind/bert-base-portuguese-cased'

# Load Dataframe:
df = pd.read_csv("./Data/FinalDataPt/allData3ClassPt.csv",usecols=['text', 'rating'])

NumberFolds = 2
NumberOverall = 53 # Pycm documentation (only int and float)
NumberClass   = 57 # Pycm documentation (only int and float) # in pycm 3.6

#%% Cross validation:
foldOverall = np.zeros((NumberFolds, NumberOverall+1)) # +1 to include AUC
foldClass   = np.zeros((NumberFolds, NumberClass, 3))
foldAUC = np.zeros((NumberFolds))
foldACC = np.zeros((NumberFolds))
foldCM = np.zeros((NumberFolds, 3, 3))
foldTime = np.zeros((NumberFolds))
best_epoch = np.zeros(NumberFolds)

# log for  final visualization
log = ['']*NumberFolds

# StratifiedKfold
skf = StratifiedKFold(n_splits=NumberFolds)

for i, (train_index, test_index) in enumerate(skf.split(df.text, df.rating)): # defaul separation train=66% and test=33%
    # Extract features and labels:
    df_train, df_val = train_test_split(df.iloc[train_index],test_size=0.3,shuffle=True)
    df_test   = df.iloc[test_index]
    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    log[i] += f"Log of fold #{i}\n"
    # Creating Data Loader
    train_data_loader = ut.create_data_loader(df_train, tokenizer, max_len, batch_size)
    val_data_loader = ut.create_data_loader(df_val, tokenizer, max_len, batch_size)
    test_data_loader = ut.create_data_loader(df_test, tokenizer, max_len, batch_size)

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
    
    # Full model to GPU
    model = SentimentClassifier(len(class_names),model_name)
    model = model.to(device)
    
    data = next(iter(train_data_loader))
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)

    # Train parametrers
    # optimizer = torch.optim.AdamW(model.parameters(), lr=l_r)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=l_r)
    
    # Scheduler
    total_steps = len(train_data_loader) * epochs
    warmup_steps = total_steps*0.1 # 10% de total_steps
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    # Loss function
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    # EarlyStop function
    early_stopping = es.EarlyStopping(verbose=True, patience=patience, optimizer=optimizer, 
                                      path=path, path_name=save_name+'Fold'+str(i), delta=delta)
        
    history = defaultdict(list)
    best_accuracy = 0
    t = time()
    print("\nTraining started...")
    
    for epoch in range(epochs):
        
        print(f'Fold {i}: Epoch {epoch + 1}/{epochs}')
        # Train function:
        train_acc, train_loss, train_auc = ut.train_epoch(model,train_data_loader,loss_fn,optimizer, 
                                                          device,scheduler,len(df_train))
        print(f'Train:  loss {train_loss:.3f}   accuracy {train_acc:.3f}')
        
        # Validation function:
        val_acc, val_loss, val_auc = ut.eval_model(model,val_data_loader,loss_fn, 
                                                   device,len(df_val))
        print(f'Val:    loss {val_loss:.3f}   accuracy {val_acc:.3f}')        
        
        history['train_acc'].append(train_acc.item())
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc.item())
        history['val_acc'].append(val_acc.item())
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc.item())
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            print(f'\n Best Accuracy: {best_accuracy.item():.3f} ')
        
        # early stopping
        early_stopping(val_auc.item(), model)
        if early_stopping.early_stop:
           print("\n Stop at epoch:", epoch+1, "\n")
           break
       
    foldTime[i] = time()-t
    best_epoch[i] = early_stopping.best_epoch
    print(f"Training finished...(Elapsed time: {strftime('%H:%M:%S', gmtime(foldTime[i]))})")
    
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
    load_path= path + 'checkpoint_' + save_name + 'Fold' + str(i) +'.pth'
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save the best model
    if save_model:
        model_path = create_dir(path[:-1], f"Model_{save_name}")
        torch.save({'epoch': epoch,
               'model_state_dict': model.state_dict(),
               'acc': val_acc}, path_model + 'model' + save_name + str(i) + '.pth')
        print("\nFull Model" + str(i) + "Save...")
        
    print("\nTesting model...\n")
    
    # Predictions:
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
    plt.ylabel('True Sentiment')
    plt.xlabel('Predicted Sentiment')
    plt.xticks(np.arange(3)+0.5, class_names)
    plt.yticks(np.arange(3)+0.5, class_names)
    plt.show()

    foldCM[i] = cmTest.to_array(normalized=True)
    
    log[i] += f" Training time: {strftime('%H:%M:%S', gmtime(foldTime[i]))}\n"
    log[i] += f" Best epoch: {best_epoch[i]}\n"
        
    # compute AUC_ROC:
    auroc = AUROC(num_classes=num_class)
    ovr_AUC = auroc(y_pred_probs, y_test).item()
    foldAUC[i] = ovr_AUC
        
    # Store metrics (assuming python dictionary is ordered)
    LabelsOverall = [k for k,v in cmTest.overall_stat.items() if isinstance(v,(float,int)) ]
    LabelsClass   = [k for k,v in cmTest.class_stat.items() if isinstance(v[0],(float,int))]
    
    foldOverall[i] = selectMetrics(cmTest.overall_stat,LabelsOverall) + [ovr_AUC]
    foldClass[i]   = np.asarray([list(d.values()) for d in selectMetrics(cmTest.class_stat,LabelsClass)])
    ACC_id = 0 # index of overall ACC (checked)
    foldACC[i] = foldOverall[i,ACC_id]

    log[i] += f" Fold Accuracy: {foldACC[i]}\n"
    log[i] += f" Classification report \n{classTest}\n"

FinalBestEpoch = best_epoch
FinaltrainTime = np.mean(foldTime)
FinalVariationAUC = np.std(foldAUC)
FinalVariationACC = np.std(foldACC)
FinalOverall = {k:v for k,v in zip( LabelsOverall+['AUC'], np.mean(foldOverall,axis=0) )}
FinalClass = {k:{j:v for j,v in enumerate(varray)} for k,varray in zip( LabelsClass, np.mean(foldClass,axis=0) )}
FinalACC = foldACC  
FinalAUC = foldAUC 
FinalCM = np.mean(foldCM, axis=0) 

# Print Log:
for f in range(NumberFolds):
    print(log[f],sep='\n')
    with open(f'{path_model}log_cv.txt','a') as log_file:
        log_file.write(log[f])
    if f == (NumberFolds-1):
        with open(f'{path_model}log_cv.txt','a') as log_file:
            log_file.write(f"\n Overall Accuracy: {FinalOverall['Overall ACC']*100:.2f} %")

# Print the output
print(f" Overall Accuracy: {FinalOverall['Overall ACC']*100:.2f} %")

# Save the metrics
if save_metrics:
    save(save_name, path=path,
        trainBestEpoch = [FinalBestEpoch],
        trainTime = [FinaltrainTime],
        trainLOSS = loss_hist,
        trainACC = acc_hist,
        trainAUC = auc_hist,
        pycmACC = FinalACC,
        pycmAUC = FinalAUC,
        pycmSTDAUC = FinalVariationAUC,
        pycmSTDACC = FinalVariationACC,
        pycmCM = FinalCM, # confussion matrix
        pycmOverall = FinalOverall,
        pycmClass = FinalClass) 
    
#=============================== END CODE ====================================

