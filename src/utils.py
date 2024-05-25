import os
import sys
import pickle as pkl
from src.exception import CustomException

def save_object(object, file_path):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path,'wb') as f:
            pkl.dump(object, f)
            
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "rb") as f:
            return  pkl.load(f)
        
    except Exception as e:
        raise CustomException(e, sys)