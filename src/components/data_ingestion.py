import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/Car details v3.csv')
            logging.info("Read the dataset as dataframe")

            logging.info("Start data cleaning")
            df.drop_duplicates(inplace=True)
            df['brand_model'] = df.name.apply(
                lambda x: ' '.join(x.split(' ')[0:2]).lower())
            df['max_power'] = df['max_power'].replace(' bhp', np.nan)
            df['max_power_bhp'] = df.max_power.apply(
                lambda x: float(x.split()[0]) if x is not np.nan else x)
            df['engine_cc'] = df.max_power.apply(
                lambda x: float(x.split()[0]) if x is not np.nan else x)
            df['mileage_kmpl'] = df.max_power.apply(
                lambda x: float(x.split()[0]) if x is not np.nan else x)
            df = df.drop(columns=['max_power', 'engine', 'mileage', 'name'])

            # droping rows with nan values
            df.dropna(inplace=True)

            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,
                      index=False, header=True)

            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,
                            index=False, header=True)

            logging.info('Ingestion of the data is completed')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    train_path, test_path = DataIngestion().initiate_data_ingestion()
    logging.info("Data ingestion complete.")

    logging.info("Initiate data transformation.")
    train_arr, test_arr, preprocessor_obj_path = DataTransformation(
    ).initiate_data_transformation(train_path, test_path)
    logging.info("Data transformation complete.")

    logging.info("Model training initiated.")
    model_trainer = ModelTainer()
    r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    logging.info("Model training complete.")
    print(r2_score)
