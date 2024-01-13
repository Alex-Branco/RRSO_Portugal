"""
@author: Alexandre

Notes: The EarlyStopping class in this is inspired by the ignite EarlyStopping class.
Link: https://pytorch.org/ignite/generated/ignite.handlers.early_stopping.EarlyStopping.html

"""
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, optimizer=None, path='./BERT/SentimentAnalysisClassPt/',
                 path_name='Sentiment_Analysis' , trace_func=print):
        """Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: './BERT/SentimentAnalysisClassPt/' must have the bar (/) in the end.
            path_name(str): Name to identify the chekpoint.
                            Default: 'Sentiment_Analysis'
            trace_func (function): trace print function.
                            Default: print """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_auc_max = np.Inf
        self.delta = delta
        self.path_name=path_name
        self.path = path + 'checkpoint_' + path_name + '.pth'
        self.trace_func = trace_func
        self.epoch = 0
        self.optimizer = optimizer
        self.best_epoch = None

    def __call__(self, val_auc, model):
        self.epoch += 1
        if  self.best_score is None:
            self.best_score = val_auc
            self.save_checkpoint(val_auc, model)
            # self.val_loss_min = val_loss   # if we use the loss function to stop the model
            self.val_auc_max = val_auc
        elif val_auc < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f' EarlyStop counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_auc
            self.trace_func(f' Best Value AUC: {val_auc:.3f}')
            self.save_checkpoint(val_auc, model)
            # self.val_loss_min = val_loss  # if we use the loss function to stop the model
            self.val_auc_max = val_auc
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        '''Saves model when AUC increase.'''
        if self.verbose:
            self.trace_func(f' Value AUC increased from ({self.val_auc_max:.3f} --> {val_auc:.3f})')
            self.trace_func(f' Saving checkpoint_{self.path_name}... \n')

        self.best_epoch = self.epoch
        torch.save({'epoch': self.epoch,
                   'model_state_dict': model.state_dict()}, self.path)
        self.val_auc_max = val_auc

#================================ END CODE ===================================  
