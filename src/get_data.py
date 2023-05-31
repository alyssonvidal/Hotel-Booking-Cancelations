import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import subprocess
import os
from config import Pathning

#!mkdir ~/.kagglegi 
#!cp /home/alysson/Downloads/kaggle.json /home/alysson/.kaggle/kaggle.json
#!chmod 600 /home/alysson/.kaggle/kaggle.json

api = KaggleApi()
api.authenticate()

api.dataset_download_file('jessemostipak/hotel-booking-demand', file_name='hotel_bookings.csv')

#SCRIPT_PATH = './scripts/get_data.sh'
os.makedirs(Pathning.DATA_RAW_PATH, exist_ok=True)



# # Adicionar permissão de execução ao script

subprocess.run(["chmod", "+x", str(Pathning.SCRIPTS_PATH / "get_data.sh")])
#subprocess.run(["chmod", "+x", Pathning.SCRIPTS_PATH / + "get_data.sh"])
subprocess.run(str(Pathning.SCRIPTS_PATH / "get_data.sh"), shell=True)