import pandas as pd
from datetime import datetime as dt
import os
import sys
from config import Path
from utils import Preprocessing


# if ROOT_DIR not in sys.path:
#     sys.path.append(ROOT_DIR)

data_raw = pd.read_csv(Path.DATA_RAW_PATH)

data_new, drop_ratio = Preprocessing.Treatment(data_raw)
data_new = Preprocessing.Featurization(data_new)
data_new = data_new.reset_index(drop=True)
os.makedirs(Path.DATA_PROCESSED_FOLDER, exist_ok=True)
data_new.to_csv(Path.DATA_PROCESSED_PATH, index=False)