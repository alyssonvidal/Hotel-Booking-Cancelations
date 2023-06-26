import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig

# Pipeline
from get_data import load_raw_data
from preprocessing import preprocess_data
from preparation import preparation
from train import train_model
from evaluate import evaluate


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


@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def pipeline(config: DictConfig):

    load_raw_data(config)
    print(f"{bcolors.OKCYAN}Data is loaded{bcolors.ENDC}")

    preprocess_data(config)
    print(f"{bcolors.OKCYAN}Data Preprocessing is done{bcolors.ENDC}")

    preparation(config)
    print(f"{bcolors.OKCYAN}Data Preparation is done{bcolors.ENDC}")

    train_model(config)
    print(f"{bcolors.OKCYAN}Training is done{bcolors.ENDC}")

    evaluate(config)
    print(f"{bcolors.OKCYAN}Evaluating is done{bcolors.ENDC}")

if __name__ == "__main__":
    pipeline()