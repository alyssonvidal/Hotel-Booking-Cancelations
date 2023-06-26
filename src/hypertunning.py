import optuna
import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier

import os
import pickle

# Model
from lightgbm import LGBMClassifier

# Project/Developer Libraries
import hydra
from omegaconf import OmegaConf, DictConfig

# Model Selection Tools
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

import mlflow

from preparation import load_data, encoding, SelectedFeatures
from utils import get_metrics, ConfusionMatrixPlot
import hydra

#Global Seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

from train import train_model
from preparation import load_data


#@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
df = pd.read_csv('/home/alysson/projects/Hotel-Booking-Cancelations/data/data_processed/train/train.csv')
X = df.drop('is_canceled', axis=1)
y = df.is_canceled



def objective(trial):

    #weight = round(float((y.value_counts()[0])/(y.value_counts()[1])),3)

    param_grid = {
        'objective': trial.suggest_categorical('objective', ['binary']),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'num_leaves': trial.suggest_int("num_leaves", 100, 300, step=20),
        'max_depth': trial.suggest_int("max_depth", 6, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 50),
        #'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1, weight]),
        'seed': RANDOM_SEED
    }

    model = LGBMClassifier(**param_grid)   

    number_folds = 3
    Kfold = StratifiedKFold(n_splits=number_folds, shuffle=True, random_state=RANDOM_SEED)
    y_pred = cross_val_predict(model, X, y, cv=Kfold)
    return f1_score(y, y_pred)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
best_params = study.best_params
best_score = study.best_value

rounded_params = {
            'objective': best_params['objective'],
            'boosting_type': best_params['boosting_type'],
            'num_leaves': best_params['num_leaves'],
            'max_depth': best_params['max_depth'],
            'learning_rate': round(best_params['learning_rate'], 4),
            'reg_alpha': round(best_params['reg_alpha'], 8),
            'reg_lambda': round(best_params['reg_lambda'], 6),
            'subsample_freq': best_params['subsample_freq'],
            'min_child_samples': best_params['min_child_samples']}

print(rounded_params)


# if __name__ == "__main__":
#     preparation()