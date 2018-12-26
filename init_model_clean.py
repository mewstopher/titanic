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

# Using cleaned data (with engineered features):
# test: .95
# train: .93
#something going on with using df vs X_train - need to investigate
os.listdir("../titanic/data/")
path = "../titanic/data/train_clean.csv"

def print_score(m):
    res = [m.score(X_train, y_train), m.score(X_test, y_test)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)

train = pd.read_csv(path)

train_cats(train)
df, y, nas = proc_df(train, 'Survived')
X_train, X_test, y_train, y_test = train_test_split(df, y)


reset_rf_samples()
m = RandomForestClassifier(n_estimators=60)
m.fit(df, y)

print_score(m)
# keep only important variables
X_train.shape
fi = rf_feat_importance(m, X_train)
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:30]);
to_keep = fi[fi.imp>0.005].cols; len(to_keep)
X_train = X_train[to_keep].copy()
X_test = X_test[to_keep].copy()

df = df[to_keep].copy()
df.shape
