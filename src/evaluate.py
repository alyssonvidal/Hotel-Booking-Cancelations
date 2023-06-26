import hydra
from omegaconf import OmegaConf, DictConfig
import os
import numpy as np

import pandas as pd
import pickle
from preparation import load_data, encoding, SelectedFeatures
from utils import get_metrics, ConfusionMatrixPlot

@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def evaluate(config: DictConfig):
    test_df = load_data(config.test.path)
    X_test = test_df.drop('is_canceled', axis=1)
    y_test = test_df.is_canceled    

    with open(config.model.path, 'rb') as file:
        model = pickle.load(file)


    y_prob = model.predict_proba(X_test)

    y_prob = y_prob[:, 1]
    y_pred = np.where(y_prob >= config.crossvalidation.threshold, 1, 0)

    ## Reports Metrics ##               
    os.makedirs(config.reports.metrics.test.dir, exist_ok=True)
    scores = get_metrics(y_test, y_pred, y_prob, config.reports.metrics.test.scores.path, save=True)

    ## Reports Plots ##   
    os.makedirs(config.reports.plots.train.dir, exist_ok=True)
    ConfusionMatrixPlot(y_test, y_pred, config.reports.plots.test.confusion_matrix.path, save=True)

    print(f'Test Scores: {scores}')
    print(f'Model used: {config.model.path}')


if __name__ == "__main__":
    evaluate()
    