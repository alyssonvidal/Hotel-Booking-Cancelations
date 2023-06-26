from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, roc_auc_score, f1_score
import json

def ConfusionMatrixPlot(y, yhat, config, save=False):
        labels = ["Não Cancelado", "Cancelado"]
        cm = confusion_matrix(y, yhat)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format="d")

        if save == True:
            plt.savefig(config, dpi=120)


def get_metrics(y_true, y_pred, y_pred_prob, config, save=False):    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    entropy = log_loss(y_true, y_pred_prob)

    scores = {'Accuracy': round(acc, 3), 
            'Precision': round(prec, 3), 
            'Recall': round(recall, 3), 
            'F1': round(f1, 3),
            'AUC': round(auc, 3),
            'Entropy': round(entropy, 3)}

    if save == True:
            with open(config, "w") as file:
                json.dump(scores, file)

    return  scores