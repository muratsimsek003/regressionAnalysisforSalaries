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


data = pd.read_excel("dataset/generatorAnalysis.xlsx")

data.head()
data.info()
data.describe()
data.isnull().sum()
data = data.dropna()
data.duplicated().sum()
data.head()
columns=data.columns

outlier_indices=detect_outliers(data, columns)

len(outlier_indices)
data= data.drop(outlier_indices,axis = 0).reset_index(drop = True)

y=data["GenYuk"]
X=data.drop(["GenYuk"],axis=1)

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
