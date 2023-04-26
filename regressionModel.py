#Python OOP yapısı kullanılmıştır.


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor


from sklearn import metrics,preprocessing, model_selection
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold,train_test_split,cross_val_score,ShuffleSplit,GridSearchCV,RandomizedSearchCV
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

class RegressionModel:
    def __init__(self, X,y):
        self.X=X
        self.y=y
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.33,random_state=42)


    def LinearRegression(self):
        print("*********Linear Regression*********")
        linear_model = LinearRegression()
        linear_model.fit(self.X_train, self.y_train)
        y_pred = linear_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print('MSE:', mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        print('MAE:', mae)
        rmse = math.sqrt(mse)
        print('RMSE:', rmse)
        r2 = r2_score(self.y_test, y_pred)
        print('R-squared:', r2)

    def RidgeRegression(self):
        print("*********Ridge Regression*********")
        ridge_model = Ridge()
        ridge_model.fit(self.X_train, self.y_train)
        y_pred = ridge_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print('MSE:', mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        print('MAE:', mae)
        rmse = math.sqrt(mse)
        print('RMSE:', rmse)
        r2 = r2_score(self.y_test, y_pred)
        print('R-squared:', r2)



    def LassoRegression(self):
        print("*********Lasso Regression*********")
        lasso_model = Lasso()
        lasso_model.fit(self.X_train, self.y_train)
        y_pred = lasso_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print('MSE:', mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        print('MAE:', mae)
        rmse = math.sqrt(mse)
        print('RMSE:', rmse)
        r2 = r2_score(self.y_test, y_pred)
        print('R-squared:', r2)



    def SVMRegression(self):
        print("*********Support Vector Regression*********")
        svmreg_model = SVR()
        svmreg_model.fit(self.X_train, self.y_train)
        y_pred = svmreg_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print('MSE:', mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        print('MAE:', mae)
        rmse = math.sqrt(mse)
        print('RMSE:', rmse)
        r2 = r2_score(self.y_test, y_pred)
        print('R-squared:', r2)


    def KNNRegression(self):
        print("*********KNN Regression*********")
        KNNreg=KNeighborsRegressor()
        KNNreg.fit(self.X_train, self.y_train)
        y_pred = KNNreg.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print('MSE:', mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        print('MAE:', mae)
        rmse = math.sqrt(mse)
        print('RMSE:', rmse)
        r2 = r2_score(self.y_test, y_pred)
        print('R-squared:', r2)


    def DecisionTreeRegression(self):
        print("*********Decision Tree Regression*********")
        dtree_model = DecisionTreeRegressor()
        dtree_model.fit(self.X_train, self.y_train)
        y_pred = dtree_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print('MSE:', mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        print('MAE:', mae)
        rmse = math.sqrt(mse)
        print('RMSE:', rmse)
        r2 = r2_score(self.y_test, y_pred)
        print('R-squared:', r2)


    def RandomForestRegressor(self):
        print("**** Random Forest Regression")
        rfr_model = RandomForestRegressor()
        rfr_model.fit(self.X_train, self.y_train)
        y_pred = rfr_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print('MSE:', mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        print('MAE:', mae)
        rmse = math.sqrt(mse)
        print('RMSE:', rmse)
        r2 = r2_score(self.y_test, y_pred)
        print('R-squared:', r2)


    def XGBRegression(self):
        print("**********XGBOOST Regression*********")
        xgb_reg=XGBRegressor()
        xgb_reg.fit(self.X_train, self.y_train)
        y_pred = xgb_reg.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print('MSE:', mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        print('MAE:', mae)
        rmse = math.sqrt(mse)
        print('RMSE:', rmse)
        r2 = r2_score(self.y_test, y_pred)
        print('R-squared:', r2)


    def GBMRegressor(self):
        print("**********GBM Regression*********")
        gb_reg = GradientBoostingRegressor()
        gb_reg.fit(self.X_train, self.y_train)
        y_pred = gb_reg.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print('MSE:', mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        print('MAE:', mae)
        rmse = math.sqrt(mse)
        print('RMSE:', rmse)
        r2 = r2_score(self.y_test, y_pred)
        print('R-squared:', r2)


    def LightGBMRegressor(self):
        print("**********LGBM Regression*********")
        lgb_reg=LGBMRegressor()
        lgb_reg.fit(self.X_train, self.y_train)
        y_pred = lgb_reg.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print('MSE:', mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        print('MAE:', mae)
        rmse = math.sqrt(mse)
        print('RMSE:', rmse)
        r2 = r2_score(self.y_test, y_pred)
        print('R-squared:', r2)


    def MLPCRegressor(self):
        print("**********MLPC Regression*********")
        mlp_reg = MLPRegressor()
        mlp_reg.fit(self.X_train, self.y_train)
        y_pred = mlp_reg.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print('MSE:', mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        print('MAE:', mae)
        rmse = math.sqrt(mse)
        print('RMSE:', rmse)
        r2 = r2_score(self.y_test, y_pred)
        print('R-squared:', r2)










