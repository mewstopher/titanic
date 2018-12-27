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
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
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

def print_score(m):
    res = [m.score(X_train, y_train), m.score(X_test, y_test)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)

path = "../titanic/output/"
data = pd.read_csv(path + 'train_cleaned.csv')

#train_cats(data)
df, y, nas = proc_df(data, 'Survived')
df = df.drop(['Age_na'], axis=1)
lr_best = LogisticRegression(C=.01)
rf_best = RandomForestClassifier(n_estimators=40)
gb_best = GradientBoostingClassifier(n_estimators=150, learning_rate=.01)
xg_best = XGBClassifier(n_estimators=40, subsample=.6)
votingC = VotingClassifier(estimators=[('lr', lr_best), ('rf', rf_best),
    ('gb',gb_best), ('xg', xg_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(df, y)


test_clean = pd.read_csv(path+"/test_cleaned.csv")
test_clean = test_clean.drop(['Survived'], axis=1)
new = votingC.predict(test_clean)
new = pd.DataFrame(new)
test = pd.read_csv("/users/brendenleavitt/documents/code/titanic/data/test.csv")
sub = pd.concat([new, test.PassengerId], axis=1)
sub.columns = ['Survived', 'PassengerId']
sub = sub.astype(int)
sub.to_csv("/users/brendenleavitt/documents/code/titanic/output/sub5.csv", index=False)
