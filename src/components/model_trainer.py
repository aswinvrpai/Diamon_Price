import pandas as pd
import numpy as np
import os,sys
from dataclasses import dataclass
import yaml

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model

# Model Trainer Libraries
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

@dataclass
class ModelTrainerconfig:
    model_training_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_training_config = ModelTrainerconfig()
        
    def intitiate_model_training(self,train_arr,test_arr):
        
        # Log;
        logging.info('Model Training initiated')
        
        try:
            
            # Retrieve Train and Test data
            X_train,y_train,X_test,y_test = [
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            ]
            
            # ML Models;
            models = {
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'RandomForestRegressor':RandomForestRegressor()
            }
            
            # Log;
            logging.info('Model Evaluation initiated')
            
            # Get the current directory of the script
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Navigate two folders up
            two_folders_up = os.path.join(current_dir, '..', '..')
            
            # Yaml File;
            yaml_file = os.path.join(two_folders_up,'hyperparam.yaml')
            
            # Hyperparameter Tuning Parameters from yaml file;
            with open(yaml_file,'r') as file_obj:
                params = yaml.safe_load(file_obj)
            
                     
            model_report = evaluate_model(X_train,y_train,X_test,y_test,models)
            
            # Log;
            logging.info('Model Evaluation completed')
            
            # Log
            logging.info(model_report)
            
            # Best Model Name and R2 score;
            best_model_name = max(model_report, key=model_report.get)
            max_value = model_report[best_model_name]
            
            best_model = models[best_model_name]
            
            # Log
            logging.info(f"Best Model - {best_model_name}, R2 Score - {max_value}")
            
            save_object(
                file_path=self.model_training_config.model_training_path,
                obj=best_model
            )
            
            # Log 
            logging.info(f"Best Model - {best_model_name}, Model Saved")
            
        except Exception as e:
            
            # Log;
            logging.info('Error occured in Model Training')
            raise CustomException(e,sys)

