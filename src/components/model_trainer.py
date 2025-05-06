import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig:
         trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
         def __init__(self):
                self.model_trainer_config=ModelTrainerConfig()
         
         def initiate_model_trainer(self,train_array,test_array):
                  try:
                          logging.info("Split training and test input data")
                          x_train,y_train,x_test,y_test = (
                                  train_array[:,:-1],
                                  train_array[:,-1],
                                  test_array[:,:-1],
                                  test_array[:,-1],
                          )
                          models = {
                           "Linear Regression": LinearRegression(),
                           "Lasso": Lasso(),
                           "Ridge": Ridge(),
                           "K-Neighbours Regressor": KNeighborsRegressor(),
                           "Decision Tree": DecisionTreeRegressor(),
                           "Random Forest": RandomForestRegressor(),
                           "XG Boost": XGBRegressor(),
                           "CatBoosting Regressor": CatBoostRegressor(),
                           "AdaBoost Regressor": AdaBoostRegressor()
                           }
                          params={
                                        "Decision Tree": {
                                        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                                
                                        },
                                        "Random Forest":{
                                        'n_estimators': [8,16,32,64,128,256]
                                        },
                                        "Linear Regression":{},
                                        "K-Neighbours Regressor":{
                                        'n_neighbors':[5,7,9,11],
                                  
                                        },
                                        "XG Boost":{
                                        'learning_rate':[.1,.01,.05,.001],
                                        'n_estimators': [8,16,32,64,128,256]
                                        },
                                        "CatBoosting Regressor":{
                                        'depth': [6,8,10],
                                        'iterations': [30, 50, 100]
                                        },
                                        "AdaBoost Regressor":{
                                        'learning_rate':[.1,.01,0.5,.001],
                                        'n_estimators': [8,16,32,64,128,256]
                                        },
                                        "Lasso": {},
                                        "Ridge": {},     
                                }
                          model_report:dict = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)
                          
                          # To get best model score from dict
                          best_model_score = max(sorted(model_report.values()))
                          
                          # To get best model name
                          best_model_name = list(model_report.keys())[
                                  list(model_report.values()).index(best_model_score)
                          ]

                          best_model = models[best_model_name]

                          if best_model_score < 0.6:
                                  raise CustomException("No best model found")
                          logging.info(f"Best model found on both training and testing dataset")

                          save_object(
                                  file_path=self.model_trainer_config.trained_model_file_path,
                                  obj=best_model
                          )

                          predicted = best_model.predict(x_test)
                          r2_square = r2_score(y_test,predicted)
                          print(best_model_name)

                          return r2_square

                          
                  except Exception as e:
                          raise CustomException(e,sys)
                  


