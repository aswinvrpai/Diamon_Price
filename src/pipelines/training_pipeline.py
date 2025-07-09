import os
import pandas as pd
import sys

# This command is very important for running the code;
# Below command is used to modify the Python path at runtime, specifically to include a parent directory to the module search path
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + '/../../')
from src.logger import logging
sys.path.pop(1)

# Data Ingestion code;
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    
    # Data Ingestion;
    obj = DataIngestion()
    train_file,test_file = obj.intiate_data_ingestion()
    print(train_file,test_file)
    
    # Data Transformation;
    data_transformation_obj = DataTransformation()
    train_arr,test_arr,preprocessor_object_path = data_transformation_obj.data_transformation(train_file,test_file)
    
    # Model Training;
    model_training_obj = ModelTrainer()
    model_training_obj.intitiate_model_training(train_arr,test_arr)
    
