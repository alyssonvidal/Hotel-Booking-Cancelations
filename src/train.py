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
#from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Preprocessing Tools
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, OrdinalEncoder, OneHotEncoder, FunctionTransformer, LabelEncoder
#from category_encoders.count import CountEncoder

# Model Selection Tools
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, cross_validate, StratifiedKFold


# Model Evaluation Tools
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc, classification_report


from config import Path
from utils import Preparation, Plots, MachineLearning, Metrics, HyperTuning

#Global Seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



data = pd.read_csv(Path.DATA_PROCESSED_PATH)
data_prep = data.copy()
data_prep = data_prep.sample(15000)

## Data Preparation
data_prep = Preparation.Encoding(data_prep)
SELECTED_FEATURES, DROP_FEATURES, TARGET = Preparation.SelectedFeatures(data_prep)
###xxx = Preparation.Standardization(data_prep_encoded, scaler_type='MinMaxScaler')

y = data_prep[TARGET]
X = data_prep[SELECTED_FEATURES]



# Hyperparamters Tuning
hypeparamters = HyperTuning(X, y)
best_params, best_score = hypeparamters.optimize(n_trials=25)


# Cross Validation
lgbm = LGBMClassifier(**best_params)
y_pred, y_prob = MachineLearning.CrossValidationPredict(lgbm, X, y, number_folds=5, threshold=0.5)

# Scores
metrics = Metrics.get_metrics(y, y_pred, y_prob)

#Create Report Folder and Files (metrics, plots, params...)
os.makedirs(Path.PLOTS_FOLDER, exist_ok=True)
Plots.ConfusionMatrixPlot(y, y_pred, save=True)

os.makedirs(Path.METRICS_FOLDER, exist_ok=True)
with open(Path.METRICS_PATH, 'w') as file:
    metrics_json = json.dumps(metrics)
    file.write(metrics_json)

os.makedirs(Path.PARAMS_FOLDER, exist_ok=True)
with open(Path.PARAMS_PATH, 'w') as file:
    params_json = json.dumps(best_params)
    file.write(params_json)