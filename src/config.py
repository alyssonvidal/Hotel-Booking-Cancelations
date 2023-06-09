from pathlib import Path


class Path:

    HOME_DIR = Path.home()
    ROOT_DIR = Path.cwd()

    DATA_RAW_FOLDER = ROOT_DIR / "data" / "data_raw"
    DATA_RAW_PATH = ROOT_DIR / "data" / "data_raw" / "data_raw.csv"

    DATA_PROCESSED_FOLDER = ROOT_DIR / "data" / "data_processed"
    DATA_PROCESSED_PATH = ROOT_DIR / "data" / "data_processed"/ "data_processed.csv"

    MODELS_FOLDER = ROOT_DIR / "models"

    SCRIPTS_FOLDER = ROOT_DIR / "scripts"

    PLOTS_FOLDER = ROOT_DIR / "reports" / "plots"
    CONFUSION_MATRIX_PATH = ROOT_DIR / "reports" / "plots" / "confusion_matrix.png"    
    ROC_AUC_PATH = ROOT_DIR / "reports" / "plots" / "roc_auc.png"

    METRICS_FOLDER = ROOT_DIR / "reports" / "metrics"
    METRICS_PATH = ROOT_DIR / "reports" / "metrics" / "metrics.json"

    PARAMS_FOLDER= ROOT_DIR / "reports" / "params"
    PARAMS_PATH = ROOT_DIR / "reports" / "params" / "params.json"


    