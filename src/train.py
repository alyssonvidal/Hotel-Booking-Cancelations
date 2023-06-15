import pandas as pd
import numpy as np

# File / OS tools
import os

# Model
from lightgbm import LGBMClassifier


# Project/Developer Libraries
import hydra
from omegaconf import OmegaConf, DictConfig


# Model Selection Tools
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, cross_validate, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
#from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, roc_auc_score, f1_score

from lightgbm import LGBMClassifier

import mlflow

from metrics import get_metrics
from plots import ConfusionMatrixPlot

from matplotlib import pyplot as plt

#Global Seed
RANDOM_SEED = 42

def load_train(file_processed: str):
    data = pd.read_csv(file_processed)
    return data

def encoding(data: pd.DataFrame):

    hotel_dict = {'City Hotel': 0, 'Resort Hotel': 1}
    data['hotel'] = data['hotel'].map(hotel_dict)

    meal_dict = {'SC': 0, 'HB': 1, 'BB': 2, 'FB': 3}
    data['meal'] = data['meal'].map(meal_dict)

    continentes_dict = {'Unknow': -1, 'Native': 0, 'Europe': 1, 'Asia': 2, 'North America': 3, 'South America': 4, 'Oceania': 5, 'Africa': 6}
    data['continentes'] = data['continentes'].map(continentes_dict)

    market_segment_dict = {'Undefined': -1, 'Online TA': 0, 'Offline TA/TO': 1, 'Groups': 2, 'Corporate': 3, 'Direct': 4, 'Aviation': 5, 'Complementary': 6}
    data['market_segment'] = data['market_segment'].map(market_segment_dict)

    distribution_dict = {'Undefined': -1, 'TA/TO': 0, 'Direct': 1, 'Corporate': 2, 'GDS': 3}
    data['distribution_channel'] = data['distribution_channel'].map(distribution_dict)

    customer_type_dict = {'Transient': 0, 'Transient-Party': 1, 'Contract': 2, 'Group': 3}
    data['customer_type'] = data['customer_type'].map(customer_type_dict)

    data['previous_cancellations'] = data['previous_cancellations'].apply(lambda x: 2 if (x >= 2) else x)
    data['previous_bookings_not_canceled'] = data['previous_bookings_not_canceled'].apply(lambda x: 2 if (x >= 2) else x)
    data['booking_changes'] = data['booking_changes'].apply(lambda x: 2 if (x >= 2) else x)

    n = 20
    top_agents = data['agent'].value_counts().nlargest(n).index
    top_companies = data['company'].value_counts().nlargest(n).index
    data['agent'] = np.where(data['agent'].isin(top_agents), data['agent'], -1)
    data['company'] = np.where(data['company'].isin(top_companies), data['company'], -1)

    return data

def SelectedFeatures(config: DictConfig):
    return config.preparation.keep_columns, config.preparation.target


@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def train_model(config: DictConfig):

    df = load_train('./data/data_processed/train/train.csv')
    df = encoding(df)
    FEATURES, TARGET = SelectedFeatures(config)
    

    mlflow.set_experiment("Hotel")
    experiment = mlflow.get_experiment_by_name("Hotel")
    with mlflow.start_run(run_name='primeiro projeto'):

        lgbm = LGBMClassifier(**config.lgbm_params)
        X = df[FEATURES] 
        y = df[TARGET].squeeze()
        ## Cross Validation         
        Kfold = StratifiedKFold(n_splits=config.crossvalidation.number_folds, shuffle=True, random_state=RANDOM_SEED)
        y_prob = cross_val_predict(lgbm, X, y, cv=Kfold, method='predict_proba', verbose=True)
        y_prob = y_prob[:, 1]
        y_pred = np.where(y_prob >= config.crossvalidation.threshold, 1, 0)
        print(y_pred)

        scores = get_metrics(y, y_pred, y_prob)

        print(scores)

        for s in scores:
            mlflow.log_metric(s, scores[s]) 
              
        mlflow.log_params(config.lgbm_params)

        ### Confusion Matrix ### 
        os.makedirs(config.plots.dir, exist_ok=True)
        ConfusionMatrixPlot(y, y_pred, config, save=True)

        mlflow.lightgbm.log_model(lgbm, "lgbm")
                
        #mlflow.log_artifact(config.plots.confusion_matrix, 'confusion_matrix')
        mlflow.set_tags({"Tag1 ":"Digite tag1", "Tag3":"Digite Tag2"})
        
        # labels = ["Não Cancelado", "Cancelado"]
        # cm = confusion_matrix(y, y_pred)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        # disp.plot(cmap="Blues", values_format="d")
        # plt.savefig(config.plots.confusion_matrix.path, dpi=120)

        print(scores)

        #print(metrics) 



if __name__ == "__main__":
    train_model()