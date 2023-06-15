import sys

import mlflow
#from steps.download_data import load_raw_data
from get_data import load_raw_data
from prep import preprocess_data
from training import train_model

#from steps.preprocess_data import preprocess_data
#from steps.tune_model import tune_model
#from steps.train_final_model import train_model

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def pipeline():
    #mlflow.set_experiment("fraud")
    file_raw = load_raw_data()
    print(f"{bcolors.OKCYAN}Data is loaded{bcolors.ENDC}")

    file_processed = preprocess_data(file_raw, missing_thr=0.95)
    print(f"{bcolors.OKCYAN}Data is preprocessed{bcolors.ENDC}")

    #file_processed = preprocess_data(file_raw, missing_thr=0.95)
    #print(f"{bcolors.OKCYAN}Data is preprocessed{bcolors.ENDC}")

    #scores = train_model(file_processed, params=None)
    


if __name__ == "__main__":
    pipeline()