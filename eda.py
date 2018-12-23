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

# looking into variables in dataset. distributions, feature engineering, etc

os.listdir("../titanic/data/")
path = "../titanic/data/train.csv"

def print_score(m):
    res = [m.score(X_train, y_train), m.score(X_test, y_test)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)


train = pd.read_csv(path)
train_cats(train)
df, y, nas = proc_df(train, 'Survived')
df.dtypes
display_all(df.tail().T)

display_all(train.tail().T)
train.dtypes
# display missing values - age, cabin, contain a high percentage
# Embarked contains some
display_all(train.isnull().sum().sort_index()/len(train))

g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
# replace missing age values with median values relating to correlated variables
g = sns.factorplot(y="Age",x="Sex",data=train,kind="box")
g = sns.factorplot(y="Age",x="Sex",hue="Pclass", data=train,kind="box")
g = sns.factorplot(y="Age",x="Parch", data=train,kind="box")
g = sns.factorplot(y="Age",x="SibSp", data=train,kind="box")

# Use Sibsp, Parch, and Pclass to calculate median Values

index_NaN_age = list(train["Age"][train["Age"].isnull()].index)

for i in index_NaN_age:
    age_med = train["Age"].median()
    age_pred = train["Age"][((train['SibSp'] == train.iloc[i]["SibSp"]) &
        (train['Parch'] == train.iloc[i]["Parch"]) &
        (train['Pclass'] == train.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred):
        train['Age'].iloc[i] = age_pred
    else:
        train['Age'].iloc[i] = age_med




g = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 6,
    palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
train.dtypes


g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])

g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 6 ,
    palette = "muted")

g = sns.distplot(train["Fare"], color="m", label="Skewness : %.2f"%(train["Fare"].skew()))
g = g.legend(loc="best")

train["Fare"] = train["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

g = sns.distplot(train["Fare"], color="b", label="Skewness : %.2f"%(train["Fare"].skew()))

g = sns.barplot(x="Sex",y="Survived",data=train)

g = sns.factorplot(x="Pclass",y="Survived",data=train,kind="bar", size = 6,
    palette = "muted")


g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,
    size=6, kind="bar", palette="muted")

train["Embarked"] = train["Embarked"].fillna("S")

g = sns.factorplot("Pclass", col="Embarked",  data=train,
    size=6, kind="count", palette="muted")




train["Famsize"] = train['SibSp'] + train['Parch'] + 1


g = sns.distplot(train["Famsize"], color="b", label="Skewness : %.2f"%(train["Fare"].skew()))

train['Single'] = train['Famsize'].map(lambda s: 1 if s == 1 else 0)
train['SmallF'] = train['Famsize'].map(lambda s: 1 if  s == 2  else 0)
train['MedF'] = train['Famsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
train['LargeF'] = train['Famsize'].map(lambda s: 1 if s >= 5 else 0)



# make variable for title

train_title = [i.split(",")[1].split(".")[0].strip() for i in train["Name"]]
train["Title"] = pd.Series(train_title)
train["Title"].head()


g = sns.countplot(x="Title",data=train)

train["Title"] = train["Title"].replace(['Lady', 'the Countess','Countess',
    'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train["Title"] = train["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
train["Title"] = train["Title"].astype(int)


g = sns.countplot(train["Title"])
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])
g = sns.factorplot(x="Title",y="Survived",data=train,kind="bar")


train.drop(labels = ["Name"], axis = 1, inplace = True)



g = sns.factorplot(x="Famsize",y="Survived",kind='bar',data = train)

train["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in train['Cabin'] ])
g = sns.factorplot(y="Survived",x="Cabin",data=train,kind="bar",order=['A','B','C','D','E','F','G','T','X'])
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

# Create categorical values for Pclass
train["Pclass"] = train["Pclass"].astype("category")
train = pd.get_dummies(train, columns = ["Pclass"],prefix="Pc")
train.drop(labels = ["PassengerId"], axis = 1, inplace = True)

display_all(train.tail().T)
