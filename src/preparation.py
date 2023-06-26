import pandas as pd
import numpy as np
import os
import hydra
from omegaconf import OmegaConf, DictConfig
from sklearn.model_selection import train_test_split

#Global Seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data(file_processed: str):
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

def split_data(df, test_size=0.2, random_state=RANDOM_SEED):
    train_df, test_df = train_test_split(df, test_size=test_size, shuffle= True, random_state=random_state)
    return train_df, test_df




@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def preparation(config: DictConfig):    
    df = load_data(config.processed_data.path)
    df = encoding(df)
    FEATURES, TARGET = SelectedFeatures(config)
    train_df, test_df = split_data(df)

    train_df = pd.concat([train_df[FEATURES], train_df[TARGET]], axis=1)
    test_df = pd.concat([test_df[FEATURES], test_df[TARGET]], axis=1)

    os.makedirs(config.train.dir, exist_ok=True)     
    train_df.to_csv(config.train.path, index=False)

    os.makedirs(config.test.dir, exist_ok=True)     
    test_df.to_csv(config.test.path, index=False)

    print(f'Train rows/columns: {train_df.shape}')
    print(f'Test rows/columns: {test_df.shape}')


if __name__ == "__main__":
    preparation()