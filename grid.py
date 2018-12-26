
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
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# trying RandomForestClassifier, LogisticRegression, and Gradient boosting, and
# XGBClassifier
print(os.listdir("../data/"))
path = "../data/"
data = pd.read_csv(path + 'train_clean.csv')
train_cats(data)
df, y, nas = proc_df(data, 'Survived')
X_train, X_test, y_train, y_test = train_test_split(df, y)

kfold = StratifiedKFold(n_splits=10)


LR = LogisticRegression(random_state=7)
lr_pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])
lr_params = {'lr__n_jobs': [-1],
    'lr__C': [.01, .1, .5, 1]}
lr_grid = GridSearchCV(lr_pipe, param_grid = lr_params, cv=kfold, scoring='accuracy', n_jobs= 4, verbose =1)
lr_grid.fit(X_train, y_train)


rf = RandomForestClassifier(random_state=7)
rf_pipe = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(random_state=7))])
rf_params = {'rf__n_estimators': [40,80,100,150]}
rf_grid = GridSearchCV(rf_pipe, param_grid=rf_params, cv=kfold, scoring = "accuracy",n_jobs=4, verbose=1)
rf_grid.fit(X_train, y_train)

gb_pipe = Pipeline([('scaler', StandardScaler()), ('gb', GradientBoostingClassifier(random_state=7))])
gb_params = {'gb__n_estimators':[40,80,100,150],
    'gb__learning_rate': [.01, .1, .6, .9]}
gb_grid = GridSearchCV(gb_pipe, param_grid=gb_params, cv=kfold, n_jobs=4,
    scoring='accuracy', verbose=1)
gb_grid.fit(X_train, y_train)

xg_pipe = Pipeline([('scaler', StandardScaler()), ('xg', XGBClassifier(random_state=7))])
xg_params = {'xg__subsample': [.01, .1, .3, .6, .9],
    'xg__n_estimators': [40, 60, 100, 150]}
xg_grid = GridSearchCV(xg_pipe, param_grid=xg_params, cv=kfold, n_jobs =4 ,
    scoring='accuracy', verbose=1)
xg_grid.fit(X_train, y_train)

print('log reg')
print(lr_grid.best_params_)
print(lr_grid.best_score_)

print('rf')
print(rf_grid.best_params_)
print(rf_grid.best_score_)

print('GB')
print(gb_grid.best_params_)
print(gb_grid.best_score_)

print('xg')
print(xg_grid.best_params_)
print(xg_grid.best_score_)
