# File which will have all the common functionality for the project work

import os
import sys
import numpy as np
import dill
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
 

def save_object(file_path,obj):
         try:
                 dir_path = os.path.dirname(file_path)
                 os.makedirs(dir_path, exist_ok=True)

                 with open(file_path,'wb') as file_obj:
                         dill.dump(obj,file_obj)
         except Exception as e:
                 raise CustomException(e,sys)

def evaluate_model(x_train,y_train,x_test,y_test,models,params,cv=3,n_jobs=3,verbose=1,refit=False):
        try:
                report = {}
                for item in models:
                        model = models[item]
                        #Training the model
                        para=params[item]
            
                        gs = GridSearchCV(model,para,cv=cv,n_jobs=n_jobs,verbose=verbose,refit=refit)
                        gs.fit(x_train,y_train)

                        model.set_params(**gs.best_params_)
                        model.fit(x_train,y_train)

                        y_train_pred = model.predict(x_train)
                        y_test_pred = model.predict(x_test)
                        
                        train_model_score = r2_score(y_train,y_train_pred)
                        test_model_score = r2_score(y_test,y_test_pred)

                        report[item] = test_model_score

                return report


        except Exception as e:
                raise CustomException(e,sys)
 
def load_object(file_path):
        try:
                with open(file_path,"rb") as file_obj:
                        return dill.load(file_obj)
        except Exception as e:
                raise CustomException(e,sys)