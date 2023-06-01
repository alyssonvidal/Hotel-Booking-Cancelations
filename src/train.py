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
from config import Pathning

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



#Global Seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


#ROOT_DIR = os.getenv('ROOT_DIR')
#data = pd.read_csv('data/data_processed/data_processed.csv')

data = pd.read_csv(str(Pathning.DATA_PROCESSED_PATH / "data_processed.csv"))


data_prep = data.copy()
data_prep = data_prep.sample(2000)

##Encoding

## CATEGORICAL FEATURES

hotel_dict = {'City Hotel': 0,  'Resort Hotel':1}
data_prep['hotel'] = data_prep['hotel'].map(hotel_dict) 

meal_dict = {'SC': 0,  'HB': 1, 'BB': 2, 'FB':3}
data_prep['meal'] = data_prep['meal'].map(meal_dict)

continentes_dict = {'Unknow':-1,'Native': 0,'Europe': 1, 'Asia': 2, 'North America':3, 'South America':4, 'Oceania':5, 'Africa':6 }
data_prep['continentes'] = data_prep['continentes'].map(continentes_dict) 

market_segment_dict = {'Undefined':-1,'Online TA': 0,'Offline TA/TO': 1, 'Groups': 2, 'Corporate':3, 'Direct':4, 'Aviation':5, 'Complementary':6}
data_prep['market_segment'] = data_prep['market_segment'].map(market_segment_dict)

distribution_dict = {'Undefined':-1,'TA/TO': 0,'Direct': 1, 'Corporate': 2, 'GDS':3}
data_prep['distribution_channel'] = data_prep['distribution_channel'].map(distribution_dict) 

customer_type_dict = {'Transient': 0,'Transient-Party': 1, 'Contract': 2, 'Contract':3, 'Group':4}
data_prep['customer_type'] = data_prep['customer_type'].map(customer_type_dict) 


## NUMERICAL FEATURES
data_prep['previous_cancellations'] = data_prep['previous_cancellations'].apply(lambda x: 2 if (x >= 2) else x)
data_prep['previous_bookings_not_canceled'] = data_prep['previous_bookings_not_canceled'].apply(lambda x: 2 if (x >= 2) else x)
data_prep['booking_changes'] = data_prep['booking_changes'].apply(lambda x: 2 if (x >= 2) else x)

n = 20
top_agents = data_prep['agent'].value_counts().nlargest(n).index
top_companies = data_prep['company'].value_counts().nlargest(n).index
data_prep['agent'] = np.where(data_prep['agent'].isin(top_agents), data_prep['agent'], -1)
data_prep['company'] = np.where(data_prep['company'].isin(top_companies), data_prep['company'], -1)

selected_features = ['hotel',
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
                     'customer_type',
                     'adr',
                     'required_car_parking_spaces',
                     'total_of_special_requests',
                     #'reservation_status_date',
                     'people',
                     'continentes',
                     #'kids',
                     'days_stay',
                     #'foreigner'
                     #'arrival_date'
                    ]

target = 'is_canceled'

data_prep[selected_features]

rbs = RobustScaler()
for col in data_prep[selected_features]:
    data_prep[col] = rbs.fit_transform(data_prep[[col]]).squeeze()  


y = data_prep[target]
X = data_prep[selected_features]



os.makedirs('reports/images', exist_ok=True)

def CM(y, y_pred):
    labels = ["Não Cancelado", "Cancelado"]
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")  
    #plt.show()
    plt.savefig('reports/images/confusion_matrix.png', dpi=120)
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
