import pandas as pd
import sys,os
import numpy as np
from dataclasses import dataclass

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + '/../../')
from src.logger import logging
from src.exception import CustomException

# Data transformation Libraries
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Utils
from utils import save_object

@dataclass
class DataTransformationConfig:
    data_transformation_config_path = os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
    
    def create_data_transformation_object(self):
        
        try:
            # Features;
            categorical_features = ['cut', 'color', 'clarity']
            numerical_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
            
            cut_map = ['Premium','Very Good','Ideal','Good','Fair']
            cut_categories  = cut_map[::-1]
            color_categories = ['D','E','F','G','H','I','J']
            clarity_categories = ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]
            
            # Numerical Pipeline;
            numerical_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            # Categorical Pipeline;
            categorical_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )
            
            # Preprocessor;
            preprocessor=ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_features),
                ('categorical_pipeline',categorical_pipeline,categorical_features)
            ])
            
            # Log;
            logging.info('Pipeline Completed')
            
            return preprocessor
        
        except Exception as e:
            logging.info('Error occured in Data Transformation Object creation')
            raise CustomException(e,sys)
        
    def data_transformation(self,train_data_file,test_data_file):
        
        # Log;
        logging.info('Data Transformation started')
        
        try:
        
            train_df = pd.read_csv(train_data_file)
            test_df = pd.read_csv(test_data_file)
            
            # Log;
            logging.info('Data Transformation - Read Train and Test completed')
            
            target_column_name = 'price'
            drop_columns = [target_column_name,'id']

            # Train Data - Features into independent and dependent features;
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            # Test Data - Features into independent and dependent features;
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            # Preprocessor Object;
            preprocessor_obj = self.create_data_transformation_object()
            
            # Log
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            # Apply the transformation;
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            # Getting Train and Test array after transformation;
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # Log
            logging.info("Preprocessing completed. Saving Pickle file started.")
            
            # Save Pickle file;
            save_object(
                file_path=self.data_transformation_config.data_transformation_config_path,
                obj=preprocessor_obj
            )

            # Log;
            logging.info('Processsor pickle created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config
            )
        
        except Exception as e:
            logging.info('Error occured in Data Transformation')
            raise CustomException(e,sys)
            
        

