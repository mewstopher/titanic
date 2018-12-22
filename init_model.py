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

# intial modeling using random forest for a base estimate
# no preprocessing done
# results give R2 values of:
# train score: .98
# test score: .82
# not terrible, but also not good(should rank around top 30% of leaderboard)
# Overfitting seems to be an issue

os.listdir("../titanic/data/")
path = "../titanic/data/train.csv"

def print_score(m):
    res = [m.score(X_train, y_train), m.score(X_test, y_test)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)

train = pd.read_csv(path)
train_cats(train)
df, y, nas = proc_df(train, 'Survived')
X_train, X_test, y_train, y_test = train_test_split(df, y)

r = RandomForestClassifier(n_estimators=60)
r.fit(X_train, y_train)
print_score(r)
