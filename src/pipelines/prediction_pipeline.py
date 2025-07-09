import pandas as pd
import numpy as np
import os,sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

class Prediction:
    def __init__(self) -> None:
        pass
    
    def predict(self,features):
        try:
            
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')
            
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            data_scaled = preprocessor.transform(features)
            result = model.predict(data_scaled)
            
            return result
            
        except Exception as e:
            
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,carat:float,
                 depth:float, 
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:float,
                 color:float,
                 clarity:float) -> None:
        self.carat = carat
        self.depth = carat
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity
    
    def get_data_as_dataframe(self):
        try:
            dict_data = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            
            dataframe = pd.DataFrame(dict_data)
            
            # Log;
            logging.info('Custom dataframe created')
            
            return dataframe        
        except Exception as e:
            raise CustomException(e,sys)