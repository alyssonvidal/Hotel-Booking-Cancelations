from pathlib import Path


# class Path:

#     HOME_DIR = Path.home()
#     ROOT_DIR = '/home/alysson/projects/Hotel-Booking-Cancelations/'

#     DATA_RAW_FOLDER = str(ROOT_DIR) + "data/data_raw"
#     DATA_RAW_PATH = str(ROOT_DIR) + "data/data_raw/data_raw.csv"

#     DATA_PROCESSED_FOLDER = ROOT_DIR / "data" / "data_processed"
#     DATA_PROCESSED_PATH = ROOT_DIR / "data" / "data_processed"/ "data_processed.csv"

#     DATA_TRAIN_FOLDER = ROOT_DIR / "data" / "data_processed" / "train"
#     DATA_TRAIN_PATH = ROOT_DIR / "data" / "data_processed"/  "train" / "train.csv"

#     DATA_TEST_FOLDER = ROOT_DIR / "data" / "data_processed" / "test"
#     DATA_TEST_PATH = ROOT_DIR / "data" / "data_processed"/ "test" / "test.csv"

#     MODELS_FOLDER = ROOT_DIR / "models"

#     SCRIPTS_FOLDER = ROOT_DIR / "scripts"

#     PLOTS_FOLDER = ROOT_DIR / "reports" / "plots"
#     CONFUSION_MATRIX_PATH = ROOT_DIR / "reports" / "plots" / "confusion_matrix.png"    
#     ROC_AUC_PATH = ROOT_DIR / "reports" / "plots" / "roc_auc.png"

#     METRICS_FOLDER = ROOT_DIR / "reports" / "metrics"
#     METRICS_PATH = ROOT_DIR / "reports" / "metrics" / "metrics.json"

#     PARAMS_FOLDER= ROOT_DIR / "reports" / "params"
#     PARAMS_PATH = ROOT_DIR / "reports" / "params" / "params.json"

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