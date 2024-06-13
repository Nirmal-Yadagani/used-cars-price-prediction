import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, find_best_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    model_dict = {"Linear Regression": LinearRegression(),
                  "Lasso": Lasso(),
                  "Ridge": Ridge(),
                  "K-Neihbors Regressor": KNeighborsRegressor(),
                  "Decision Tree": DecisionTreeRegressor(),
                  "Random Forest Regressor": RandomForestRegressor(),
                  "XGBRegressor": XGBRegressor(),
                  "Catboosting Regressor": CatBoostRegressor(verbose=False),
                  "AdaBoost Regressor": AdaBoostRegressor()
                  }


class ModelTainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data.")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            model_report = evaluate_models(
                models_dict=self.model_trainer_config.model_dict, X_train=X_train,
                y_train=y_train, X_test=X_test, y_test=y_test)

            best_model_name, best_model_score = find_best_model(
                models_report=model_report, metric='r2_score')

            best_model = self.model_trainer_config.model_dict[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found.")
            logging.info('Best model found on both train and test dataset')

            logging.info('save model.pkl file to artifacts.')
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        object=best_model)

            r2_score = model_report[best_model_name]['r2_score_test']

            return r2_score

        except Exception as e:
            raise CustomException(e, sys)
