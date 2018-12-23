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
path = "../titanic/data/train.csv"

def print_score(m):
    res = [m.score(X_train, y_train), m.score(X_test, y_test)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)


data = pd.read_csv(path)
train_cats(data)


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
dat = multi_median(data, 'Age', 'SibSp', 'Parch','Pclass')

path = "../titanic/data/test.csv"
test = pd.read_csv(path)


train["Famsize"] = train['SibSp'] + train['Parch'] + 1

train['Single'] = train['Famsize'].map(lambda s: 1 if s == 1 else 0)
train['SmallF'] = train['Famsize'].map(lambda s: 1 if  s == 2  else 0)
train['MedF'] = train['Famsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
train['LargeF'] = train['Famsize'].map(lambda s: 1 if s >= 5 else 0)



# make variable for title
train_title = [i.split(",")[1].split(".")[0].strip() for i in train["Name"]]
train["Title"] = pd.Series(train_title)
train["Title"] = train["Title"].replace(['Lady', 'the Countess','Countess',
    'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train["Title"] = train["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
train["Title"] = train["Title"].astype(int)
train.drop(labels = ["Name"], axis = 1, inplace = True)
train["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in train['Cabin'] ])
train = pd.get_dummies(train, columns = ["Cabin"],prefix="Cabin")

## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X.
Ticket = []
for i in list(train.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
train["Ticket"] = Ticket
train = pd.get_dummies(train, columns = ["Ticket"], prefix="T")

train["Pclass"] = train["Pclass"].astype("category")
train = pd.get_dummies(train, columns = ["Pclass"],prefix="Pc")
train.drop(labels = ["PassengerId"], axis = 1, inplace = True)
