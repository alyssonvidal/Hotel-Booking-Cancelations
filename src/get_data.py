import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import subprocess
import os
from config import Path

#!mkdir ~/.kagglegi 
#!cp /home/alysson/Downloads/kaggle.json /home/alysson/.kaggle/kaggle.json
#!chmod 600 /home/alysson/.kaggle/kaggle.json

api = KaggleApi()
api.authenticate()

api.dataset_download_file('jessemostipak/hotel-booking-demand', file_name='hotel_bookings.csv')

os.makedirs(Path.DATA_RAW_FOLDER, exist_ok=True)

#Adicionar permissão de execução ao script

subprocess.run(["chmod", "+x", str(Path.SCRIPTS_FOLDER / "get_data.sh")])
subprocess.run(str(Path.SCRIPTS_FOLDER / "get_data.sh"), shell=True)