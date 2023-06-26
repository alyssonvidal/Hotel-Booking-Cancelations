import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig
from zipfile import ZipFile

@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def load_raw_data(config: DictConfig):

    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
            raise Exception("Kaggle API key not found.")

    os.makedirs(config.raw_data.dir, exist_ok=True)
    kaggle.api.dataset_download_files('jessemostipak/hotel-booking-demand', path=config.raw_data.dir)

    zip_file = os.path.join(config.raw_data.dir, "hotel-booking-demand.zip")

    with ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(config.raw_data.dir)

    
    #Rename File from hotel_bookings.csv to data_raw.csv
    old_name = os.path.join(config.raw_data.dir, "hotel_bookings.csv")
    new_name = os.path.join(config.raw_data.dir, "data_raw.csv")
    os.rename(old_name, new_name)
    
    os.remove(zip_file)

    print(f'Data raw path: {new_name}') 


if __name__ == "__main__":
    load_raw_data()