import pandas as pd
import datetime
import os
import hydra
from omegaconf import OmegaConf, DictConfig

import numpy as np
import pycountry
import pycountry_convert as pc

def process_nans(df: pd.DataFrame, drop_thr: float) -> pd.DataFrame: #=0.95
    for col in df.columns:
        nulls_prop = df[col].isnull().mean()
        print(f"{col}: {100*nulls_prop:.6f}% missing")
        
        # Drop if missing more than a threshold
        if nulls_prop >= drop_thr:
            print("Dropping", col)
            df = df.drop(col, axis=1)
        # If some values are missing
        elif nulls_prop > 0:
            print("Imputing", col)
            # If numeric impute
            nan_replacements = {"children": 0, "agent": 0, "company": 0, "country":"Unknow", "country_name":"Unknow", "continentes":"Unknow"}
            df = df.fillna(nan_replacements)

    return df


def process_stranges(df):
    init = len(df)
    # Drop no people booking
    df = df.loc[~((df['children'] == 0) & (df['adults'] == 0) & (df['babies'] == 0) & (df['reservation_status'] == 'Check-Out'))]
    # Drop negative ADR
    df = df.loc[~(df['adr'] < 0)]
    end = len(df)
    print(f"Total Dropped Strange Rows: {init-end}")

    return df

def process_duplicated(df):
    init = len(df)
    # Drop Duplicated and keep last record
    df = df.drop_duplicates(keep='last')
    end = len(df)
    print(f"Total Dropped Duplicated Rows: {init-end}")

    return df


def process_static(df):
    # Drop static columns
    static_cols = df.columns[df.nunique() == 1]
    print("Dropping static columns:", static_cols)
    df = df.drop(static_cols, axis=1)

    return df


def process_featurization(df):
    # Create Some Features
    df['meal'] = df['meal'].replace("Undefined", "SC")

    df["people"] = (df["adults"] + df["children"] + df["babies"])
    df['kids'] = df['children'] + df['babies']
    df['days_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

    # Create Country Name and Comtinent Column
    country_names = []
    for country_code in df['country']:
        try:
            country_name = pycountry.countries.get(alpha_3=country_code).name
        except AttributeError:
            country_name = 'Unknow'
        except LookupError:
            country_name = 'Unknow'
        country_names.append(country_name)
    df['country_name'] = country_names

    continents = []
    for country in df['country_name']:
        try:
            country_code = pc.country_name_to_country_alpha2(country)
            continent_name = pc.country_alpha2_to_continent_code(country_code)
            continent_code = pc.convert_continent_code_to_continent_name(continent_name)
            continents.append(continent_code)
        except:
            continents.append(None)
    df['continentes'] = continents

    df.loc[df['country'] == 'PRT', 'continentes'] = 'Native'

    print('Features Created: meal, people, kids, days_stay, country_name, continentes')

    return df


@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def preprocess_data(config: DictConfig):
    data_raw = pd.read_csv(config.raw_data.path)
    df = process_nans(data_raw, config.raw_data.missing_thr)
    df = process_stranges(df) 
    df = process_duplicated(df)
    df = process_static(df)    
    df = process_featurization(df)

    os.makedirs(config.processed_data.dir, exist_ok=True)
    df.to_csv(config.processed_data.path, index=False)
    

if __name__ == "__main__":
    preprocess_data()