###########3

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pickle
import numpy as np


app = FastAPI()

class InputData(BaseModel):
    # Defina aqui os campos correspondentes aos dados de entrada do seu modelo
    # Certifique-se de que os tipos correspondem aos dados de entrada do modelo
    # Neste exemplo, estou assumindo que você tem uma matriz 2D de características  - hotel
  hotel : float
  lead_time : float
  arrival_date_year : float
  arrival_date_week_number : float
  meal : float
  market_segment : float
  distribution_channel : float
  is_repeated_guest : float
  previous_cancellations : float
  previous_bookings_not_canceled : float
  booking_changes : float
  agent : float
  company : float
  customer_type : float
  adr : float
  required_car_parking_spaces : float
  people : float
  days_stay : float
  continentes : float


 #   features: List[List[float]]

MODEL_PATH = '/home/alysson/projects/Hotel-Booking-Cancelations/models/lgbm/lgbm_23-06-23.pkl'

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Krish Youtube Channel': f'{name}'}

@app.post("/predict")
def predict(data: InputData):
    # Converte os dados de entrada para o formato esperado pelo modelo
    X = np.array([[data.hotel, data.lead_time, data.arrival_date_year, data.arrival_date_week_number, data.meal,
                   data.market_segment, data.distribution_channel, data.is_repeated_guest,
                   data.previous_cancellations, data.previous_bookings_not_canceled, data.booking_changes,
                   data.agent, data.company, data.customer_type, data.adr, data.required_car_parking_spaces,
                   data.people, data.days_stay, data.continentes]])
    
    # Faz a previsão usando o modelo
    y_prob = model.predict_proba(X)
    y_prob = y_prob[:, 1]
    y_pred = np.where(y_prob >= 0.5, 1, 0)

    if y_pred==1:
        print('Provavel cancelamento')
        prediction = 'Provavel Cancelamento'
    else:
        print('De boas')
        prediction = 'Provavel De Boas'    
    
    # Retorna as previsões como resultado da API
    return {"predictions": y_pred.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)