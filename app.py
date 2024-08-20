import streamlit as st
import pandas as pd

from src.utils import load_object
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Define the Streamlit UI
st.title('Used Car Price Predictor')

# Input fields for numerical features
st.sidebar.header('Input Features')

# Example numerical features
year = st.sidebar.slider('Year of Purchase', 1960, 2024, step=1)
km_driven = st.sidebar.slider('Kilometers Driven',0,200000,step=5)
seats = st.sidebar.slider('Seater',1,10,step=1)
max_power_bhp = st.sidebar.slider('Max BHP',10,1000,step=10)
engine_cc = st.sidebar.slider('CC',100,1000,step=10)
mileage_kmpl = st.sidebar.slider('Mileage (in km)', 1, 100, step=5)

# Example categorical features
cat_features = load_object('artifacts\categories_dict.pkl')
transmission = st.sidebar.selectbox('Transmission', cat_features['transmission'])
owner = st.sidebar.selectbox('Owner',cat_features['owner'])
fuel = st.sidebar.selectbox('Fuel',cat_features['fuel'])
seller_type = st.sidebar.selectbox('Seller type',cat_features['seller_type'])
brand_model = st.sidebar.selectbox('Brand model',cat_features['brand_model'])

# Create a DataFrame from user inputs
data = CustomData(year=year, km_driven=km_driven, fuel=fuel, seller_type=seller_type, transmission=transmission, owner=owner, seats=seats, brand_model=brand_model, max_power_bhp=max_power_bhp, engine_cc=engine_cc, mileage_kmpl=mileage_kmpl)

# Predict the price
if st.sidebar.button('Predict Price'):
    model = PredictPipeline()
    prediction = model.predict(data.get_data_as_dataframe())
    st.write(f'Predicted Price: ${prediction[0]:.2f}')

