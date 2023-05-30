import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import subprocess

#!mkdir ~/.kaggle
#!cp /home/alysson/Downloads/kaggle.json /home/alysson/.kaggle/kaggle.json
#!chmod 600 /home/alysson/.kaggle/kaggle.json

api = KaggleApi()
api.authenticate()

api.dataset_download_file('jessemostipak/hotel-booking-demand', file_name='hotel_bookings.csv')

SCRIPT_PATH = './scripts/get_data.sh'

# Adicionar permissão de execução ao script
subprocess.run(["chmod", "+x", SCRIPT_PATH])
subprocess.run(SCRIPT_PATH, shell=True)