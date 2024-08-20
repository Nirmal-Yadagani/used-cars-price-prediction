import sys
import os

import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            tranasformed_data = preprocessor.transform(features)
            prediction = model.predict(tranasformed_data)

            return prediction
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, 
                 year: int, 
                 km_driven: int,
                 fuel: str,
                 seller_type: str, 
                 transmission:str, 
                 owner: str,	
                 seats: int,	
                 brand_model: str, 
                 max_power_bhp: int,
                 engine_cc: int, 
                 mileage_kmpl: int):
        self.year = year
        self.km_driven = km_driven
        self.fuel = fuel
        self.seller_type = seller_type 
        self.transmission = transmission 
        self.owner = owner
        self.seats = seats
        self.brand_model = brand_model
        self.max_power_bhp = max_power_bhp
        self.engine_cc = engine_cc
        self.mileage_kmpl = mileage_kmpl

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {"year" : [self.year], 
                                      "km_driven": [self.km_driven], 
                                      "fuel" : [self.fuel],
                                      "seller_type": [self.seller_type],
                                      "transmission": [self.transmission],
                                      "owner": [self.owner],
                                      "seats": [self.seats],
                                      "brand_model": [self.brand_model],
                                      "max_power_bhp":[self.max_power_bhp],
                                      "engine_cc":[self.engine_cc],
                                      "mileage_kmpl":[self.mileage_kmpl]}
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == '__main__':
    data = CustomData(year=1988,km_driven=73980,fuel='CNG',seller_type='Dealer',transmission='4.0',owner='First Owner',seats=5,brand_model='ambassador classic',max_power_bhp=430,engine_cc=490,mileage_kmpl=41)
    predict_pipeline = PredictPipeline()
    prediction = predict_pipeline.predict(data.get_data_as_dataframe())
    print(prediction[0])