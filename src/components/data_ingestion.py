import sys
import os
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd

@dataclass
class DataIngestionconfig():
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')

class DataIngestion():
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionconfig()

    def intiate_data_ingestion(self):
        logging.info('Data Ingestion Starts')

        try:
            
            # Write the raw data to CSV;
            df = pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path)
            logging.info('Data Ingestion to Raw data file completed')

            # Train Test Split and write out to train and test split file;
            train_set,test_set = train_test_split(df,test_size=0.3,random_state=42)

            # Train set data to csv;
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info('Data Ingestion to Train data file completed')

            # Test set data to csv;
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Data Ingestion to Test data file completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info('Error occurred at Dataingestion')
            raise CustomException(e,sys)
