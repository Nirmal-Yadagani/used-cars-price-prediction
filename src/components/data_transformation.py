import os
import sys
from src.utils import save_object
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, OrdinalEncoder,  MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
import numpy as np

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        logging.info("Create data transformation pipeline.")
        try:
            # numerical columns
            num_cols = ['km_driven','year','max_power_bhp','engine_cc','mileage_kmpl']

            # Catagorical columns
            ordinal_encoded = ['transmission','owner']
            target_encoded = ['seats']
            one_hot_binary = ['fuel','seller_type']
            one_hot_multiclass = ['brand_model']

            # Create transformation pipeline
            target_encoded_pipeline = Pipeline([
                ('target_encoding', TargetEncoder(smooth=0.5,target_type="continuous"))
            ])

            num_pipeline = Pipeline([
                ('minmax_scaler', MinMaxScaler())    
            ])

            ordinal_encoded_pipeline = Pipeline([
                ('ordinal_encoding',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-2))
            ])

            onehot_encoded_pipeline = Pipeline([
                ('one_hot_encoding', OneHotEncoder(sparse_output=False,handle_unknown='ignore'))
            ])

            onehot_multiclass_pipeline = Pipeline([
                ('one_hot_multiclass', OneHotEncoder(max_categories=50,sparse_output=False,handle_unknown='ignore'))
            ])

            preprocessor_ct = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_cols),
                ('target_encoder', target_encoded_pipeline, target_encoded),
                ('ordinal_encoder', ordinal_encoded_pipeline, ordinal_encoded),
                ('one_hot_encoder', onehot_encoded_pipeline, one_hot_binary),
                ('one_hot_multiclass', onehot_multiclass_pipeline, one_hot_multiclass)
            ])

            return preprocessor_ct

        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("read train and test data completed.")

            logging.info("obtaining preprocessing object.")
            preprocessor_object = self.get_data_transformer_object()

            # target column
            target_column_name = 'selling_price'

            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_features_train_df = train_df[target_column_name]
            target_features_test_df = test_df[target_column_name]

            logging.info("applying preprocessor object on train and test date.")
            input_features_train_array = preprocessor_object.fit_transform(input_features_train_df, target_features_train_df)
            input_features_test_array = preprocessor_object.transform(input_features_test_df, target_features_test_df)

            train_arr = np.c_[input_features_train_array, np.array(target_features_train_df)]
            test_arr = np.c_[input_features_test_array, np.array(target_features_test_df)]

            logging.info("save preprocessor.pkl file to artifacts.")
            save_object(object=preprocessor_object,file_path=self.transformation_config.preprocessor_obj_file_path)

            return train_arr, test_arr, self.transformation_config.preprocessor_obj_file_path
        
        except Exception as e:
            raise CustomException(e, sys)