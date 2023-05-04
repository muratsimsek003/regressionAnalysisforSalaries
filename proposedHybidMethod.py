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


data = pd.read_csv('used_device_data.csv')
data.isnull().sum()
data = data.dropna()
data.os.value_counts()
columns=data.columns
from labelEncoderColumns import labelEncoderColumns
data_le=labelEncoderColumns(data)
data_le.head()
from detect_outliers import detect_outliers
outlier_indices=detect_outliers(data_le, columns)
len(outlier_indices)

data_le= data_le.drop(outlier_indices,axis = 0).reset_index(drop = True)

data_le.info()

dummies_year = pd.get_dummies(data_le['release_year'], prefix='year', drop_first=True)

data = pd.concat([data_le,dummies_year],axis=1)
data.head()
data = data.drop('release_year',axis=1)
data.head()
data.to_csv("dataset/normalizephone.csv")
y = data['normalized_used_price']
X = data.drop(['normalized_used_price'], axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

####Ridge Regression GridSearchCV
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
# list of alpha to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
#initialising Ridge() function
ridge = Ridge()
ridge_cv_model = GridSearchCV(estimator=ridge,param_grid=params,scoring='neg_mean_squared_error',cv=5,verbose=0)
ridge_cv_model.fit(X_train,y_train)
ridge_best = ridge_cv_model.best_estimator_
ridge_best
ridge_tuned = ridge_best.fit(X_train,y_train)
y_pred_ridge = ridge_tuned.predict(X_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
mse = mean_squared_error(y_test, y_pred_ridge)
print('MSE:', mse)
mae = mean_absolute_error(y_test, y_pred_ridge)
print('MAE:', mae)
rmse = math.sqrt(mse)
print('RMSE:', rmse)
r2 = r2_score(y_test, y_pred_ridge)
print('R-squared:', r2)


from sklearn.ensemble import  GradientBoostingRegressor
gb_reg = GradientBoostingRegressor()
gb_reg.fit(X_train, y_train)
y_pred_gbm = gb_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred_gbm)
print('MSE:', mse)
mae = mean_absolute_error(y_test, y_pred_gbm)
print('MAE:', mae)
rmse = math.sqrt(mse)
print('RMSE:', rmse)
r2 = r2_score(y_test, y_pred_gbm)
print('R-squared:', r2)


grid = dict()
grid['n_estimators'] = [10, 50, 100, 500]
grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
grid['subsample'] = [0.5, 0.7, 1.0]
grid['max_depth'] = [3, 7, 9]
from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=gb_reg, param_grid=grid, n_jobs=-1, cv=cv)
grid_search.fit(X, y)

y_pred_tuned_gbm = grid_search.predict(X_test)
mse = mean_squared_error(y_test, y_pred_tuned_gbm)
print('MSE:', mse)
mae = mean_absolute_error(y_test, y_pred_tuned_gbm)
print('MAE:', mae)
rmse = math.sqrt(mse)
print('RMSE:', rmse)
r2 = r2_score(y_test, y_pred_tuned_gbm)
print('R-squared:', r2)

from sklearn.model_selection import KFold



# model_1 = Ridge(alpha=0.5)
# model_2 = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

# Initialize the voting Ensemble Learning model
voting_model = VotingRegressor([('ridge', ridge_tuned), ('random_forest', grid_search)])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the voting Ensemble Learning model on the training set
voting_model.fit(X_train, y_train)

# Make predictions on the test set using the voting Ensemble Learning model
y_pred = voting_model.predict(X_test)

# Compute the root mean squared error of the voting Ensemble Learning model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE of the voting Ensemble Learning model:", rmse)

mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)
mae = mean_absolute_error(y_test, y_pred)
print('MAE:', mae)
rmse = math.sqrt(mse)
print('RMSE:', rmse)
r2 = r2_score(y_test, y_pred)
print('R-squared:', r2)