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

dict_G = {'yes':1,'no':0}

data['4g'] = data['4g'].map(dict_G)

data['5g'] = data['5g'].map(dict_G)

data.head()

data.os.value_counts()

sns.countplot(data, x="os")
plt.show()

data.device_brand.value_counts()

data = data.drop(['os','device_brand'],axis=1)

data.head()

data['release_year'].value_counts()

data.hist(figsize=(12,8),bins=30)
plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(12,8))
sns.heatmap(data.corr(),annot=True)
plt.show()

columns=data.columns



outlier_indices=detect_outliers(data, columns)

len(outlier_indices)

data = data.drop(outlier_indices,axis = 0).reset_index(drop = True)

data.info()

dummies_year = pd.get_dummies(data['release_year'], prefix=['year'], drop_first=True)

data = pd.concat([data,dummies_year],axis=1)
data.head()
data = data.drop('release_year',axis=1)
data.head()

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