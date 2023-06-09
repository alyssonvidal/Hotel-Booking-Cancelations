import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from config import Path

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, roc_auc_score, f1_score
from matplotlib import pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier

import optuna

#Global Seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class Preparation:
    @staticmethod
    def Encoding(data: pd.DataFrame):
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
    
    @staticmethod
    def SelectedFeatures(data: pd.DataFrame):
        SELECTED_FEATURES = ['hotel',
                             #'is_canceled',
                             'lead_time',
                             'arrival_date_year',   
                             #'arrival_date_month',
                             'arrival_date_week_number',
                             #'arrival_date_day_of_month',
                             #'stays_in_weekend_nights',                    
                             #'stays_in_week_nights',
                             #'adults',
                             #'children',
                             #'babies',
                             'meal',
                             #'country',
                             'market_segment',
                             'distribution_channel',
                             'is_repeated_guest',
                             'previous_cancellations',
                             #'assigned_room_type',
                             'previous_bookings_not_canceled',
                             #'reserved_room_type',
                             'booking_changes',
                             #'deposit_type',
                             'agent',
                             'company',
                             #'days_in_waiting_list',
                             'customer_type']
        
        TARGET = 'is_canceled'

        DROP_FEATURES = [value for value in data if value != 'is_canceled' and value not in data[SELECTED_FEATURES]]

        return SELECTED_FEATURES, DROP_FEATURES, TARGET
    

    @staticmethod
    def Standardization(data: pd.DataFrame, type: str):
        if type == 'RobustScaler':
            scaler = RobustScaler()
        elif type == 'MinMaxScaler':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Scaler type must be either 'RobustScaler' or 'MinMaxScaler'")

        for col in data:
            data[col] = scaler.fit_transform(data[[col]]).squeeze()

        return data
    

class Plots:
    @staticmethod
    def ConfusionMatrixPlot(y, yhat, save=False):
        labels = ["Não Cancelado", "Cancelado"]
        cm = confusion_matrix(y, yhat)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format="d")

        if save == True:
            plt.savefig(Path.CONFUSION_MATRIX_PATH, dpi=120)

        plt.show()
        #print("Score:\n", classification_report(y, yhat))


from sklearn.model_selection import StratifiedKFold, cross_val_predict
import numpy as np

class MachineLearning:
    @staticmethod
    def CrossValidationPredict(model, X, y, number_folds=5, threshold=0.5):
        Kfold = StratifiedKFold(n_splits=number_folds, shuffle=True, random_state=RANDOM_SEED)

        y_prob = cross_val_predict(model, X, y, cv=Kfold, method='predict_proba', verbose=False)
        y_prob = y_prob[:, 1]

        y_pred = np.empty(shape=(len(y_prob)))
        for i in range(len(y_prob)):
            if y_prob[i] >= threshold:
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        return y_pred, y_prob


class Metrics:
    @staticmethod
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
                'Auc': round(auc, 3),
                'Entropy': round(entropy, 3)}
    
class HyperTuning:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def objective(self, trial):

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
        y_pred = cross_val_predict(model, self.X, self.y, cv=Kfold)
        return f1_score(self.y, y_pred)

    def optimize(self, n_trials):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)

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

        return rounded_params, best_score