import os
import sys

import pickle as pkl
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_squared_error

from src.exception import CustomException


def save_object(object, file_path):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path, 'wb') as f:
            pkl.dump(object, f)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "rb") as f:
            return pkl.load(f)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(models_dict, X_train, y_train, X_test, y_test):
    report = {}
    for model_name, model in models_dict.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        r2Score_train = r2_score(y_train, y_pred_train)
        r2Score_test = r2_score(y_test, y_pred_test)
        root_mean_squared_error_train = root_mean_squared_error(
            y_train, y_pred_train)
        root_mean_squared_error_test = root_mean_squared_error(
            y_test, y_pred_test)
        mean_absolute_error_train = mean_absolute_error(y_train, y_pred_train)
        mean_absolute_error_test = mean_absolute_error(y_test, y_pred_test)
        mean_squared_error_train = mean_squared_error(y_train, y_pred_train)
        mean_squared_error_test = mean_squared_error(y_test, y_pred_test)

        report[model_name] = {'r2_score_train': r2Score_train,
                              'r2_score_test': r2Score_test,
                              'root_mean_squared_error_train': root_mean_squared_error_train,
                              'root_mean_squared_error_test': root_mean_squared_error_test,
                              'mean_absolute_error_train': mean_absolute_error_train,
                              'mean_absolute_error_test': mean_absolute_error_test,
                              'mean_squared_error_train': mean_squared_error_train,
                              'mean_squared_error_test': mean_squared_error_test}

    return report


def find_best_model(models_report, metric):
    metric = metric + '_test'
    best_model = ""
    best_model_score = 0
    for model_name, model_scores in models_report.items():
        if best_model_score < model_scores[metric]:
            best_model_score = model_scores[metric]
            best_model = model_name

    return best_model, best_model_score
