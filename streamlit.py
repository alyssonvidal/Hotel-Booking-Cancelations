import streamlit as st
import pandas as pd
import pickle
from src.preparation import encoding


st.set_page_config(page_title='Hotel Booking Cancelations', layout='wide')

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

else:
    hotel = st.selectbox('Hotel',('City Hotel','Resort Hotel'))
    lead_time = st.slider('Lead Time (days)', 1,360,30)
    arrival_date_year = st.selectbox('Year', list(reversed(range(2015,2018))))
    arrival_date_week_number = st.selectbox('Week of Year', list(reversed(range(1,54))))
    meal = st.selectbox('Meal',('HB','BB','FB'))            
    market_segment = st.selectbox('Market Segment',('Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Complementary', 'Groups', 'Undefined', 'Aviation'))
    distribution_channel = st.selectbox('Distribution Channel',('Direct', 'Corporate', 'TA/TO', 'Undefined', 'GDS'))
    is_repeated_guest = st.selectbox('Repeated',(1,0))
    previous_cancellations = st.number_input('Total Previous Cancelation', 0, step=1)
    previous_bookings_not_canceled = st.number_input('Total Previous Booking Not Canceled', 0, step=1)
    booking_changes = st.number_input('Total Booking Changes', 0, step=1)
    agent = st.number_input('Number of Agency', 0, step=1)
    company = st.number_input('Number of Company', 0, step=1)
    customer_type = st.selectbox('Customer Type',('Transient', 'Contract', 'Transient-Party', 'Group'))
    adr = st.number_input('Total ADR', 0)
    required_car_parking_spaces = st.number_input('Number of Parking Spaces Requiered', 0, step=1)
    people = st.number_input(('Number of people per room'), 1, step=1)
    days_stay = st.slider('Total Days Stay', 1,60,2)
    continentes = st.selectbox('Continent',('Native', 'Europe', 'North America', 'Asia', 'South America', 'Oceania', 'Africa'))         
                    

    dict = {'hotel' : hotel,
            'lead_time': lead_time,
            'arrival_date_year' : arrival_date_year,
            'arrival_date_week_number' : arrival_date_week_number,
            'meal': meal,
            'market_segment' : market_segment,
            'distribution_channel' : distribution_channel,
            'is_repeated_guest' : is_repeated_guest,
            'previous_cancellations' : previous_cancellations,
            'previous_bookings_not_canceled' : previous_bookings_not_canceled,
            'booking_changes' : booking_changes,
            'agent' : agent,
            'company' : company,
            'customer_type' : customer_type,
            'adr' : adr,
            'required_car_parking_spaces' : required_car_parking_spaces,
            'people' : people,
            'days_stay' : days_stay,
            'continentes' : continentes}
    
    df = pd.DataFrame(dict, index=[0])

    results = df.copy()

    df_new = encoding(df)

    load_clf = pickle.load(open('./models/lgbm/lgbm_23-06-23.pkl', 'rb'))
    prediction = load_clf.predict(df_new)
    prediction_proba = load_clf.predict_proba(df_new)[:,1]
    results['predcit_proba'] = prediction_proba

    st.subheader('Prediction')
    st.write(results)

    

    st.markdown( '''
    **Credit:** App built in `Python` + `Streamlit` by [Alysson Machado](https://www.linkedin.com/in/alyssonmach/).
    ''')

#if __name__ == "__main__": 
