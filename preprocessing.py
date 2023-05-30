import pandas as pd
from datetime import datetime as dt
import os
import sys
#import dotenv
import pycountry
import pycountry_convert as pc


#dotenv.load_dotenv(dotenv.find_dotenv())
#ROOT_DIR = os.getenv('ROOT_DIR')

# ROOT_DIR = '/home/alysson/projects/Hotel-Booking-Cancelations'

# if ROOT_DIR not in sys.path:
#     sys.path.append(ROOT_DIR)

# os.chdir(ROOT_DIR)

# DATA_RAW_PATH = os.path.join(ROOT_DIR, "data", "data_raw", "data_raw.csv")
DATA_RAW_PATH = 'data/data_raw/data_raw.csv'
data_raw = pd.read_csv(DATA_RAW_PATH)

data = data_raw.copy()

# Replace NaN values
nan_replacements = {"children": 0, "agent": 0, "company": 0}
data = data.fillna(nan_replacements)

# Convert String format to Datetime format YYYY-MM-DD
data['reservation_status_date'] = pd.to_datetime(data['reservation_status_date'])

# Fixing Data Types
data = data.astype({"children": int, "agent": int, "company": int})

# Removing strange values: No person, Negative ADR
data=data.loc[~((data['children'] == 0) & (data['adults'] == 0) & (data['babies'] == 0) & (data['reservation_status'] == 'Check-Out'))]
data = data.loc[~(data['adr']<0)]

#Feature Engiennier
data['meal'] = data['meal'].replace("Undefined", "SC")
data["people"] = (data["adults"] + data["children"] + data["babies"])

data['kids'] = data['children'] + data['babies']
data['days_stay'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']

def add_country_names(df):
    country_names = []
    for country_code in df['country']:
        try:
            country_name = pycountry.countries.get(alpha_3=country_code).name
        except AttributeError:
            country_name = None
        except LookupError:
            country_name = None
        country_names.append(country_name)
    df['country_name'] = country_names
    return df


def add_continent(df):
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

    return df


data = add_country_names(data)
data = add_continent(data)
data.loc[data['country'] == 'PRT', 'continentes'] = 'Native'

nan_replacements = {"children": 0, "agent": 0, "company": 0, "country":"Unknow", "country_name":"Unknow", "continentes":"Unknow"}
data = data.fillna(nan_replacements)


data=data.reset_index(drop=True)

#data.to_csv(f"{ROOT_DIR}/data/data_processed/data_processed.csv", index=False)
data.to_csv('data/data_processed/data_processed.csv', index=False)