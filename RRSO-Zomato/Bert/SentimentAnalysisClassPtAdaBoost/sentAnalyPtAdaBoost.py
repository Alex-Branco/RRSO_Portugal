"""
Author: Alexandre

Note: this script uses adaboost from paper Multi-class adaboos -> zhu et.al 2006  -> SAMME

    SAMME - Stagewise Additive Modeling using a Multi-class Exponential loss function
"""
# Import libraries:
import torch
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from torch import nn
from torchmetrics import AUROC
from pycm import ConfusionMatrix
from transformers import logging
from time import time, strftime, gmtime
from sklearn.metrics import classification_report
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup

# Libraries created localy:
from EarlyStop import earlyStopAdaboost as es_ada
from PytorchTools import utilsAdaboost as uta
from PytorchTools.getMetrics import show_train_history
from PytorchTools.mySaveMetrics import selectMetrics, save, create_dir

# To ignore deprecated warnings:
warnings.filterwarnings("ignore")
# BERT warning --> this warning means that during your training, you're not 
# using the pooler in order to compute the loss
logging.set_verbosity_error()

# set CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Constants:
freeze_bert = False # freeze BERT model to train only the classifier

# Number of weak classifiers:
NumberIterations = 2  # number of weak classifiers

# Save name for metrics and checkpoint:
save_name = 'SentAnalyPtAdaBoost_DANIEL_' + str(NumberIterations) # change the name to avoid overwrite
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
path = './NewBert/SentimentAnalysisClassPtAdaBoost/'
# set path for save the model:
path_model = path + 'Model_' + save_name + '/'

# Language
lang = 'Pt'
num_class = 3
data_path = './Data/FinalData' + lang + '/'
class_names = ['Negative', 'Neutral', 'Positive']

# Set the name to import the model:
model_name = 'neuralmind/bert-base-portuguese-cased'

#%% Load Dataset:
# Load data 80% of dataset for training
df_train = pd.read_csv(data_path + '/trainData_' + str(num_class) + 'Class_'+lang +'.csv', index_col=0)
# Load 70% of the 20% remaining of dataset for valifation
df_val = pd.read_csv(data_path + '/valData_' + str(num_class) + 'Class_'+lang +'.csv', index_col=0)
# Load 30% of the 20% remaining of dataset for testing
df_test = pd.read_csv(data_path + '/testData_' + str(num_class) + 'Class_'+lang +'.csv', index_col=0)

# Load Tokenizer:
tokenizer = BertTokenizer.from_pretrained(model_name) # BertTokenizer

# Creating Data Loader:
train_data_loader = uta.create_data_loader(df_train, tokenizer, max_len, batch_size)
val_data_loader = uta.create_data_loader(df_val, tokenizer, max_len, batch_size)
test_data_loader = uta.create_data_loader(df_test, tokenizer, max_len, batch_size)    

# df_train.shape, df_val.shape, df_test.shape # show the shape of the datasets

#%% AdaBoost Functions: 
# this is the implementation os SAMME - Stagewise Additive Modeling using a Multi-class Exponential loss function

def compute_error(y, y_pred, w_i):
    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)

def compute_alpha(error, num_class):  # this is for multi class (named SAMME — Stagewise Additive Modeling using a Multi-class Exponential loss function)
    return np.log((1 - error) / error) + np.log(num_class-1)

def update_weights(w_i, alpha, y, y_pred):
    wi = w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))
    weights = wi / wi.sum()  # Renormalize 
    return weights

#%% AdaBoost 
# Clear before calling
alphas = [] 
training_errors = []
w_i_update = []

y_pred_adaBoost = []
y_pred_vote_final = []
y_test_vote_final = []

# para depois ter as probabilidades do treino e do teste para ensemble voting
y_pred_adaBoost_prob = []
y_test_vote_pred_final_prob =[]

classifier_acc = []
ovr_AUC_wsv = []

# Set weights for current boosting iteration
w_i = np.ones(len(df_train)) * 1 / len(df_train)

adaTime = np.zeros((NumberIterations))
best_epoch = np.zeros((NumberIterations))

# log for  final visualization
log = ['']*NumberIterations

# Iterate over NumberIterations weak classifiers
for i, m in enumerate(range(0, NumberIterations)): # not using m
    log[i] += f"Classifier #{i} " + save_name + "\n"
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

    # Fit weak classifier and predict labels  
    model = SentimentClassifier(len(class_names),model_name)
    model = model.to(device)
    
    data = next(iter(train_data_loader))
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)

    # Train parameters optimizer:
    # optimizer = torch.optim.AdamW(model.parameters(), lr=l_r) 
    optimizer = torch.optim.Adagrad(model.parameters(), lr=l_r) # this is the correct

    # Scheduler function: #!!! possivel retirar (colocar depois na função de treino)
    # total_steps = len(train_data_loader) * epochs
    # warmup_steps = total_steps*0.1 # 10% de total_steps
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, 
    #                                             num_training_steps=total_steps)

    # Loss function:
    loss_fn = nn.CrossEntropyLoss(reduction='none').to(device)
  
    # Early Stopping:
    early_stopping = es_ada.EarlyStopping(verbose=True, patience=patience, optimizer=optimizer, 
                                      path=path, path_name=save_name+'Ada'+str(i), delta=delta)
    
    history = defaultdict(list)
    best_accuracy = 0
    t = time()
    print("\nTraining started...")
    
    for epoch in range(epochs):
        print(f'BERT {i}: Epoch {epoch + 1}/{epochs}')
        # Train function: #!!! colocar o sheduler a entrar no treino
        train_acc, train_loss, train_auc = uta.train_epoch(model, train_data_loader, loss_fn, optimizer, device,
                                                            len(df_train), w_i, batch_size)
        print(f'Train:  loss {train_loss:.3f}   accuracy {train_acc:.3f}')
        
        # Validation function:
        val_acc, val_loss, val_auc = uta.eval_model(model, val_data_loader, loss_fn, 
                                                   device, len(df_val))
        print(f'Val:    loss {val_loss:.3f}   accuracy {val_acc:.3f}')
        
        # Save metrics histoty:
        history['train_acc'].append(train_acc.item())
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc.item())
        history['val_acc'].append(val_acc.item())
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc.item())
      
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            print(f'\n Best Accuracy: {best_accuracy.item():.3f} ')
            
        # Early stopping
        early_stopping(val_auc.item(),val_acc, model)
        if early_stopping.early_stop:
           print("\n Stop at epoch:", epoch+1, "\n")
           break
    
    adaTime[i] = time()-t
    best_epoch[i] = early_stopping.best_epoch
    print(f"Training finished...(Elapsed time: {strftime('%H:%M:%S', gmtime(adaTime[i]))})")
    
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
    load_path = path + 'checkpoint_' + save_name + 'Ada' + str(i) +'.pth'
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    classifier_acc.append(checkpoint['acc'].detach().cpu().item())
    
    # Save the classifier model 
    if save_model:
        model_path = create_dir(path[:-1], f"Model_{save_name}")
        torch.save({'epoch': epoch,
               'model_state_dict': model.state_dict(),
               'acc': val_acc}, path_model + 'model' + save_name + 'Ada' + str(i) + '.pth')
        print("\nClassifier Ada "+ str(i) + " Model Save...")
    
    print("\nPredicting...")
    # Predictions for Boosting:    
    y_review_texts_ada, y_pred_ada, y_pred_probs, y_train = uta.get_predictions(model, train_data_loader)
    y_pred_adaBoost.append(y_pred_ada.numpy())
    y_pred_adaBoost_prob.append(y_pred_probs.numpy())
    
    log[i] += f" Training time: {strftime('%H:%M:%S', gmtime(adaTime[i]))}\n"
    log[i] += f" Best epoch: {best_epoch[i]}\n"
    log[i] += f" Accuracy: {classifier_acc[i]}\n"
    
    print("\nComputing error...")
    # Compute error
    error_m = compute_error(y_train.numpy(), y_pred_ada.numpy(), torch.from_numpy(w_i))
    training_errors.append(error_m)
    # Compute alpha 
    alpha_m = compute_alpha(error_m, num_class)
    alphas.append(alpha_m)
    # Update weights
    w_i = update_weights(w_i, alpha_m.numpy(), y_train.numpy(), y_pred_ada.numpy())
    w_i_update.append(w_i)
    print("Updating weights...\n")
    
    # Predictions for voting 
    y_review_texts_vote, y_pred_vote, y_pred_probs_vote, y_test_vote = uta.get_predictions(model, test_data_loader)
    y_pred_vote_final.append(y_pred_vote.numpy()) 
    y_test_vote_final.append(y_test_vote.numpy()) 
    y_test_vote_pred_final_prob.append(y_pred_probs_vote.numpy())
    
    # compute AUC_ROC of each classifier:
    auroc_wsv = AUROC(num_classes=num_class)
    ovr_AUC_wsv.append(auroc_wsv(y_pred_probs_vote, y_test_vote).item())
    log[i] += f" AUC value: {ovr_AUC_wsv[i]}\n"

#%% weighted soft voting:
    
# normalize the accuracy values to obtain the weights per classifier
weights = [acc / sum(classifier_acc) for acc in classifier_acc]

# calculate the weighted average of the classifier predictions
ensemble_final_pred_wsv = np.average(y_test_vote_pred_final_prob, axis=0, weights=weights)

# the final ensemble prediction is the class with the highest probability
final_pred_wsv = np.argmax(ensemble_final_pred_wsv, axis=1)
# print(f'final predictions: {final_pred_wsv}')

#%% Final Predictions:
# Classification report:
classTest_wsv = classification_report(y_test_vote, final_pred_wsv, target_names=class_names)
# print(classTest_wsv) # visualização teste

cmTest_wsv = ConfusionMatrix(actual_vector= y_test_vote.tolist(), predict_vector = final_pred_wsv.tolist())
# print (f'\n === Confusion Matrix ===\n  \n {cmTest_wsv}')  # visualização teste

# Confusion Matrix:
cmTest_wsv.plot(cmap=plt.cm.Blues, number_label=True,
            normalized=True,    # Using normalized because dataset in imbalanced
            plot_lib="seaborn",
            title="Confusion Matrix")
plt.ylabel('True Sentiment')
plt.xlabel('Predicted Sentiment')
plt.xticks(np.arange(3)+0.5, class_names)
plt.yticks(np.arange(3)+0.5, class_names)
plt.show()


#%% Store metrics:
# Names of metrics to store from PyCM
LabelsOverall = [k for k,v in cmTest_wsv.overall_stat.items() if isinstance(v,(float,int)) ]
LabelsClass   = [k for k,v in cmTest_wsv.class_stat.items() if isinstance(v[0],(float,int))]

# Extract metrics from cm object
rawOverall = selectMetrics(cmTest_wsv.overall_stat,LabelsOverall)
rawClass   = np.asarray([list(d.values()) for d in selectMetrics(cmTest_wsv.class_stat,LabelsClass)])
ACC_id = 0 # index of overall ACC (checked)
FinalACC = rawOverall[ACC_id]

log[i] += f"\nFinal Accuracy: {FinalACC}\n"

# Convert into dictionary
mOverall = {k:v for k,v in zip( LabelsOverall, rawOverall )}
mClass = {k:{j:v for j,v in enumerate(varray)} for k,varray in zip( LabelsClass, rawClass )}

log[i] += f"F1 Macro: {mOverall['F1 Macro']:.3f}\n"
log[i] += f"\nClassification report: \n{classTest_wsv}\n"

FinaltrainTime = np.mean(adaTime)
FinalOverall = {k:v for k,v in zip( LabelsOverall+['AUC'], rawOverall )}
FinalClass = {k:{j:v for j,v in enumerate(varray)} for k,varray in zip( LabelsClass, rawClass )}
FinalVariationACC = np.std(classifier_acc)
FinalClassifierACC = classifier_acc
# Print Log:
for f in range(NumberIterations):
    print(log[f],sep='\n')
    with open(f'{path_model}log_'+ save_name +'.txt','a') as log_file:
        log_file.write(log[f])

# print( f" Classification report TEST\n{classTest_wsv}\n")

# Save the model:
if save_metrics:
    save(save_name, path=path,
         trainBestEpoch = [early_stopping.best_epoch],
         trainTime = [FinaltrainTime],
         trainLOSS = loss_hist,
         trainACC = acc_hist,
         trainAUC = auc_hist,
         pycmClassifierACC = FinalClassifierACC,
         pycmSTDACC = FinalVariationACC,
         pycmOverall = mOverall,
         pycmClass = mClass,
         pycmCM = cmTest_wsv.to_array(normalized=True)) # confussion matrix

#================================= END CODE ===================================
