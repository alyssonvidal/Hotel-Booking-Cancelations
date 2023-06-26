import pandas as pd
import numpy as np
import os
import pickle

# Model
from lightgbm import LGBMClassifier

# Project/Developer Libraries
import hydra
from omegaconf import OmegaConf, DictConfig

# Model Selection Tools
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from lightgbm import LGBMClassifier

import mlflow

from preparation import load_data, encoding, SelectedFeatures
from utils import get_metrics, ConfusionMatrixPlot

#Global Seed
RANDOM_SEED = 42

def load_train(file_processed: str):
    data = pd.read_csv(file_processed)
    return data

@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def train_model(config: DictConfig):

    train_df = load_data(config.train.path)
    X_train = train_df.drop('is_canceled', axis=1)
    y_train = train_df.is_canceled

    mlflow.set_experiment("Hotel")
    experiment = mlflow.get_experiment_by_name("Hotel")
    with mlflow.start_run(run_name='primeiro projeto'):

        lgbm = LGBMClassifier(**config.lgbm_params)       
        ## Cross Validation         
        Kfold = StratifiedKFold(n_splits=config.crossvalidation.number_folds, shuffle=True, random_state=RANDOM_SEED)
        y_prob = cross_val_predict(lgbm, X_train, y_train, cv=Kfold, method='predict_proba', verbose=True)
        y_prob = y_prob[:, 1]
        y_pred = np.where(y_prob >= config.crossvalidation.threshold, 1, 0)

        ## Mlflow Tracking ##        
              
        mlflow.log_params(config.lgbm_params)
        mlflow.lightgbm.log_model(lgbm, "lgbm")
        mlflow.set_tags({"Tag1 ":"Digite tag1", "Tag3":"Digite Tag2"})

        ## Reports Metrics ##               
        os.makedirs(config.reports.metrics.train.dir, exist_ok=True)
        scores = get_metrics(y_train, y_pred, y_prob, config.reports.metrics.train.scores.path, save=True)        
        for s in scores:
            mlflow.log_metric(s, scores[s]) 

        ## Reports Plots ##   
        os.makedirs(config.reports.plots.train.dir, exist_ok=True)
        ConfusionMatrixPlot(y_train, y_pred, config.reports.plots.train.confusion_matrix.path, save=True)
        #mlflow.log_artifact(config.plots.confusion_matrix, 'confusion_matrix')

        ## Save Model ##
        with open(config.model.path, 'wb') as file:
            lgbm.fit(X_train, y_train)
            pickle.dump(lgbm, file)
        
        print(f'Validation Scores: {scores}')

        print(f'Model is saved: {config.model.path}')


if __name__ == "__main__":
    train_model()