from pathlib import Path


class Pathning:
    HOME_DIR = Path.home()
    ROOT_DIR = Path.cwd()
    DATA_RAW_PATH = ROOT_DIR / "data" / "data_raw"
    DATA_PROCESSED_PATH = ROOT_DIR / "data" / "data_processed"
    MODELS_PATH = ROOT_DIR / "models"
    SCRIPTS_PATH = ROOT_DIR / "scripts"