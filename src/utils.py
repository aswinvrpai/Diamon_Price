import os
import sys
import pickle
import numpy as np 
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            # param= params[model_name]
            
            # Log;
            logging.info(f'Model Training - {model_name} Started')
            
            # grid_search = GridSearchCV(model, param_grid=param, cv=5)
            # grid_search.fit(X_train, y_train)
            
            # Best parameters for Random Forest
            # best_params = grid_search.best_params_
            
            # Train model
            # model.set_params(**best_params)
            model.fit(X_train,y_train)

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            test_model_score = r2_score(y_test,y_test_pred)
            
            # Log;
            logging.info(f'Model Training - {model_name} Finished')

            report[list(models.keys())[i]] =  test_model_score

        return report
    
    except Exception as e:
            logging.info('Exception occured during model training')
            raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)