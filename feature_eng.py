import sys
sys.path.append('/anaconda3/envs/fastai-cpu/lib/python3.6/site-packages')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier
from fastai.imports import *
from fastai.structured import *
from sklearn.metrics import mean_squared_error
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# This script applies the changes in eda in a more condensed way

os.listdir("../titanic/data/")
path = "../titanic/data/"

def print_score(m):
    res = [m.score(X_train, y_train), m.score(X_test, y_test)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)


data = pd.read_csv(path+'train.csv')
train_cats(data)
#df, y, nas = proc_df(train, 'Survived')

def multi_median(data, impute_var, *corr_vars):
    """
    finds median value depending on other correlated
    variables

    parameters
    ----------
    data: input dataframe
    impute_var: variable with missing values
    corr_vars: variables correlated with impute_var

    output
    ----------
    dataframe with median values in place of NAs
    """

    var_index = list(data[impute_var][data[impute_var].isnull()].index)
    var_med = data[impute_var].median()
    for j in var_index:
        data1 = data
        for i in corr_vars:
            var_value = data.iloc[j][i]
            data1 = data1.loc[data1[i] == var_value]
        data_out = data1[impute_var].median()

        if not np.isnan(data_out):
            data[impute_var].iloc[j] = data_out
        else:
            data[impute_var].iloc[j] = var_med
    return(data)
dataset = multi_median(data, 'Age', 'SibSp', 'Parch','Pclass')


# convert Sex into categorical value 0 for male and 1 for female
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})


# fare
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

# family size
dataset["Famsize"] = dataset['SibSp'] + dataset['Parch'] + 1
dataset['Single'] = dataset['Famsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Famsize'].map(lambda s: 1 if  s == 2  else 0)
dataset['MedF'] = dataset['Famsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Famsize'].map(lambda s: 1 if s >= 5 else 0)

# Embarked

dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")

# make variable for title
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess',
    'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)
dataset.drop(labels = ["Name"], axis = 1, inplace = True)
dataset.shape
# variable for missing cabbin values

dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")

## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X.
Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
dataset["Ticket"] = Ticket
dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")

dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")
dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)
dataset.shape
