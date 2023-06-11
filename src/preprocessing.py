import pandas as pd
from datetime import datetime as dt
import os
from config import Path
from utils import Preprocessing


#Collecting Data
data_raw = pd.read_csv(Path.DATA_RAW_PATH)
#data_raw = data_raw.sample(frac=0.2)

#Preprocessing Steps
data_new, drop_ratio = Preprocessing.Treatment(data_raw)
data_new = Preprocessing.Featurization(data_new)
data_new = data_new.reset_index(drop=True)

#Saving File Preprocessed
os.makedirs(Path.DATA_PROCESSED_FOLDER, exist_ok=True)
data_new.to_csv(Path.DATA_PROCESSED_PATH, index=False)
print(len(data_new))