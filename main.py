import pandas as pd
import warnings
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from preprep import *
from all_methods import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("dataset/salaries.csv")

df.head(10)

df.isnull().sum()

check_df(df)

df1=df.drop(["salary","salary_currency"], axis=1)


cat_cols, num_cols, cat_but_car=grab_col_names(df1)

for col in cat_cols:
    cat_summary(df1, col, plot=True)

for col in num_cols:
    num_summary(df1, col, plot=True)

for col in num_cols:
    print(col)




df1.info()

df1.head(10)

for col in num_cols:
    print(col, check_outlier(df1, col))



sns.boxplot(x = df1.salary_in_usd)
plt.show()

low_limit, up_limit= outlier_thresholds(df1,  "salary_in_usd")

df_out=df1[~((df1["salary_in_usd"] < (low_limit)) | (df1["salary_in_usd"]> (up_limit)))]

sns.boxplot(x = df_out.salary_in_usd)
plt.show()

df_out=df1[~(df1["salary_in_usd"]> 300000)]
df_out.head()
sns.boxplot(x = df_out.salary_in_usd)
plt.show()

df_out=one_hot_encoder(df_out,cat_cols,drop_first=True)

df_out.head()
df_out["job_title"].value_counts()
df_out["company_location"].value_counts()

sns.countplot(x=df_out["company_location"], data=df_out)
plt.show()


df_usa=df_out[df_out["company_location"]=="US"]

df_usa.head()


df_usa["company_location"].value_counts()

df_usa["employee_residence"].value_counts()

df_usa=df_usa[df_usa["employee_residence"]=="US"]

df_usa.drop(["company_location","employee_residence"], axis=1)

df_usa.to_csv("dataset/usa_data.csv")

title=["Data Scientist","Data Engineer","Data Analyst","Machine Learning Engineer"]


df_usa_4=df_usa[(df_usa["job_title"]=="Data Scientist")|(df_usa["job_title"]=="Data Engineer")|
                (df_usa["job_title"]=="Data Analyst")|(df_usa["job_title"]=="Machine Learning Engineer")]


df_usa_4.head()

df_usa_4.info()


df_usa_4["job_title"].value_counts()

df_usa_4 = pd.get_dummies(df_usa_4,columns=["job_title"],prefix=["job_title"], drop_first=True)

df_usa_4.head()

df_usa_4.drop(["employee_residence","company_location"], axis=1, inplace=True)

df_usa_4.head()

df_usa_4.to_csv("dataset/4job_title_usa.csv")

y=df_usa_4["salary_in_usd"]
X=df_usa_4.drop(["salary_in_usd"],axis=1)

df_usa_4["salary_in_usd"].hist(bin=50)
plt.xlabel(df_usa_4["salary_in_usd"])
plt.title(df_usa_4["salary_in_usd"])
plt.show()

df_usa_4["salary_in_usd"].value_counts()

df_usa_data=df_usa[(df_usa["job_title"]=="Data Scientist")]
df_usa_data.head(10)

df_usa_data.drop(["employee_residence","company_location"],axis=1, inplace=True)
df_usa_data.info()
y=df_usa_data["salary_in_usd"]
X=df_usa_data.drop(["salary_in_usd"],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

all_models_df=all_models(X, y, classification=False)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
param_grid = [
 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
 ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
 scoring='neg_mean_squared_error',
return_train_score=True)
grid_search.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error
y_pred_train = grid_search.predict(X_train)
y_pred_test = grid_search.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(rmse_test)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))