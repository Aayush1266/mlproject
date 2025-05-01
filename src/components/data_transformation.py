import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

@dataclass
class DataTransformationConfig:
         preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
         '''
         This function is responsible for data Transformation
         '''
         def __init__(self):
                 self.data_transformation_config = DataTransformationConfig()
         def get_data_transformer_object(self):
                 try:
                         numerical_columns = ['writing_score','reading_score','math_score']
                         categorical_columns = [
                                 'gender',
                                 'race_ethnicity',
                                 'lunch',
                                 'parental_level_of_education',
                                 'test_preparation_course'
                         ]
                         num_pipeline = Pipeline(
                                 steps=[
                                         ("imputer",SimpleImputer(strategy='median')),
                                         ("scaler",StandardScaler())
                                 ]
                         )
                         cat_pipeline = Pipeline(
                                 steps=[
                                         ("imputer",SimpleImputer(strategy='most_frequent')),
                                         ("One_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                                         ("Scaler",StandardScaler(with_mean=False))
                                 ]
                         )

                         logging.info("Numerical columns standard scaling completed")
                         logging.info("Categorical columns encoding completed")

                         preprocessor = ColumnTransformer(
                                 [
                                         ("numerical_pipeline",num_pipeline,numerical_columns),
                                         ("categroical_pipeline",cat_pipeline,categorical_columns)
                                 ]
                         )
                         return preprocessor               

                 except Exception as e:
                         raise CustomException(e,sys)
                 
         
         def initiate_data_transformation(self,train_path,test_path):
                 try:
                         train_df = pd.read_csv(train_path)
                         test_df = pd.read_csv(test_path)

                         logging.info("Read train and test data completed")
                         logging.info("Obtaining preprocessing object")

                         preprocessing_obj = self.get_data_transformer_object()

                         target_name = 'total_score'
                         numerical_columns = ['writing_score','reading_score','math_score']
                         input_feature_train_df = train_df.drop(columns=[target_name],axis=1)
                         target_feature_train_df = train_df[target_name]

                         input_feature_test_df = test_df.drop(columns=[target_name],axis=1)
                         target_feature_test_df = test_df[target_name]

                         logging.info("Applying preprocessing object on Training Dataframe and test dataframe")
                         input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                         input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

                         train_arr = np.c_[
                                 input_feature_train_arr, np.array(target_feature_train_df)
                         ]
                         test_arr = np.c_[
                                 input_feature_test_arr, np.array(target_feature_test_df)
                         ]

                         logging.info(f"Saving preprocessing objects")

                        # function to save the pickle file for preprocessing object
                         save_object(
                                 file_path = self.data_transformation_config.preprocessor_obj_file_path,
                                 obj=preprocessing_obj
                         )

                         return (
                                 train_arr,
                                 test_arr,
                                 self.data_transformation_config.preprocessor_obj_file_path
                         )

                 except Exception as e:
                         raise CustomException(e,sys)