import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, roc_auc_score, f1_score



def get_metrics(y_true, y_pred, y_pred_prob):    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    entropy = log_loss(y_true, y_pred_prob)
    return {'Accuracy': round(acc, 3), 
            'Precision': round(prec, 3), 
            'Recall': round(recall, 3), 
            'F1': round(f1, 3),
            'AUC': round(auc, 3),
            'Entropy': round(entropy, 3)}  
