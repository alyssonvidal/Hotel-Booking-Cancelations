# Basic Tools
import pandas as pd
import numpy as np

# File / OS tools
import json
import os

# Model
from lightgbm import LGBMClassifier

# Project/Developer Libraries
from config import Path
from utils import Preparation, Plots, MachineLearning, Metrics, HyperTuning

#Global Seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



data = pd.read_csv(Path.DATA_PROCESSED_PATH)
data_prep = data.copy()

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