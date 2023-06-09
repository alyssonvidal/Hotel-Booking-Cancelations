# Basic Tools
import pandas as pd
import numpy as np
from datetime import datetime as dt

# Visualization Tools
from matplotlib import pyplot as plt
import seaborn as sns

# File / OS tools
import json
import os

# Machine Learning Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Preprocessing Tools
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, OrdinalEncoder, OneHotEncoder, FunctionTransformer, LabelEncoder
#from category_encoders.count import CountEncoder

# Model Selection Tools
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, KFold, cross_validate, StratifiedKFold


# Model Evaluation Tools
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc, classification_report


from config import Path
from utils import Preparation

# if ROOT_DIR not in sys.path:
#     sys.path.append(ROOT_DIR)


#Global Seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


#ROOT_DIR = os.getenv('ROOT_DIR')
#data = pd.read_csv('data/data_processed/data_processed.csv')

data = pd.read_csv(str(Path.DATA_PROCESSED_PATHFILE))


data_prep = data.copy()
data_prep = data_prep.sample(20000)

##Encoding
data_prep = Preparation.Encoding(data_prep)



y = data_prep[target]
X = data_prep[selected_features]



os.makedirs('reports/plots', exist_ok=True)

def CM(y, y_pred):
    labels = ["Não Cancelado", "Cancelado"]
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")  
    #plt.show()
    plt.savefig('reports/plots/confusion_matrix.png', dpi=120)
    plt.close()
    #print("Score: \n", classification_report(y,y_pred))

def ROC(model, y, y_prob, model_dict):
    score_metrics_auc = pd.DataFrame(columns=['Model','AUC']) 
    fpr,tpr, threshold = roc_curve(y,y_prob)
    auc = roc_auc_score(y,y_prob)
    plt.figure(figsize=(4, 3))
    plt.plot(fpr,tpr, color='steelblue', label = model_dict)    
    plt.title("ROC")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.plot([0,1],[0,1], color='black', linestyle='--')
    plt.xlabel("False Positives Rate (1- Specifity)")
    plt.ylabel("True Positives Rate (Sensitivity)")
    plt.legend(loc = 'lower right') 
    #plt.show()
    plt.savefig('roc_model.png', dpi=120)
    print(f"AUC: {auc:.4f}\n\n")     
   
    return y_prob, auc



lgbm = LGBMClassifier(random_state=RANDOM_SEED)

number_folds = 5

Kfold = StratifiedKFold(n_splits=number_folds, shuffle=True, random_state=RANDOM_SEED )

y_prob = cross_val_predict(lgbm, X, y, cv=Kfold, method='predict_proba', verbose=False)    
y_prob = y_prob[:,1]

### Defining threshold ###
y_pred = np.empty(shape=(len(y_prob)))
threshold = 0.5
for i in range(len(y_prob)):    
    if y_prob[i] >= threshold:
        y_pred[i] = 1  
    else:
        y_pred[i] = 0  

CM(y,y_pred)
#ROC(lgbm, y, y_prob, 'lgbm') 

scores = cross_validate(lgbm, X, y, cv = Kfold, scoring=['accuracy','precision','recall','f1','roc_auc'], return_train_score=True)

f1_train_mean = round(np.mean(scores['train_f1']),5)
f1_val_mean = round(np.mean(scores['test_f1']),5)

f1_results = {
'F1 Train': f1_train_mean,
'F1 Validation': f1_val_mean    
}

json_data = json.dumps(f1_results)

with open('reports/metrics/metrics.json', 'w') as file:
    file.write(json_data)
