import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from detect_outliers import detect_outliers
from regressionModel import RegressionModel


data = pd.read_csv('used_device_data.csv')

data.head()
data.info()
data.describe()
data.isnull().sum()
data = data.dropna()
data.duplicated().sum()
data.head()
data.os.value_counts()
columns=data.columns

from data_visualization import visualization

visualization(columns, data)

from labelEncoderColumns import labelEncoderColumns

data_le=labelEncoderColumns(data)

data_le.head()

outlier_indices=detect_outliers(data_le, columns)

len(outlier_indices)

data_le= data_le.drop(outlier_indices,axis = 0).reset_index(drop = True)

data_le.info()

dummies_year = pd.get_dummies(data_le['release_year'], prefix='year', drop_first=True)

data = pd.concat([data_le,dummies_year],axis=1)
data.head()
data = data.drop('release_year',axis=1)
data.head()

# data.to_csv("dataset/processDevicedata.csv")
# data.head()

y = data['normalized_used_price']
X = data.drop(['normalized_used_price'], axis=1)


regression=RegressionModel(X,y)

regression.LinearRegression()
regression.RidgeRegression()
regression.LassoRegression()
regression.KNNRegression()
regression.SVMRegression()
regression.DecisionTreeRegression()
regression.RandomForestRegressor()
regression.GBMRegressor()
regression.LightGBMRegressor()
regression.XGBRegression()
regression.MLPCRegressor()



import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor,VotingRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

####Ridge Regression GridSearchCV
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
# list of alpha to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
#initialising Ridge() function
ridge = Ridge()
ridge_cv_model = GridSearchCV(estimator=ridge,param_grid=params,scoring='neg_mean_squared_error',cv=5,n_jobs = -1, verbose = 2)
ridge_cv_model.fit(X_train,y_train)
ridge_best = ridge_cv_model.best_estimator_
ridge_best
ridge_tuned = ridge_best.fit(X_train,y_train)
y_pred_tuned = ridge_tuned.predict(X_test)
mse = mean_squared_error(y_test, y_pred_tuned)
print('MSE:', mse)
mae = mean_absolute_error(y_test, y_pred_tuned)
print('MAE:', mae)
rmse = math.sqrt(mse)
print('RMSE:', rmse)
r2 = r2_score(y_test, y_pred_tuned)
print('R-squared:', r2)


rf=RandomForestRegressor(random_state=42)
param_grid = {
            "n_estimators"      : [10,20,100],
            "max_features"      : ["auto", "sqrt", "log2"],
            "min_samples_split" : [2,4,8],
            "bootstrap": [True, False],
            }

grid = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1, cv=5,verbose = 2)
grid.fit(X_train, y_train)

y_pred_tuned_rf = grid.predict(X_test)
mse = mean_squared_error(y_test, y_pred_tuned_rf)
print('MSE:', mse)
mae = mean_absolute_error(y_test, y_pred_tuned_rf)
print('MAE:', mae)
rmse = math.sqrt(mse)
print('RMSE:', rmse)
r2 = r2_score(y_test, y_pred_tuned_rf)
print('R-squared:', r2)

from sklearn.model_selection import KFold




# Initialize the voting Ensemble Learning model
voting_model = VotingRegressor([('ridge', ridge_cv_model), ('random_forest', grid)])


# Fit the voting Ensemble Learning model on the training set
voting_model.fit(X_train, y_train)

# Make predictions on the test set using the voting Ensemble Learning model
y_pred = voting_model.predict(X_test)

# Compute the root mean squared error of the voting Ensemble Learning model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE of the voting Ensemble Learning model:", rmse)
