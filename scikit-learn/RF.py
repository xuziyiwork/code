import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
tree.fit(X_train, y_train)

tree.apply(X)
dp = tree.decision_path(X_test)
tree.predict(X_test)
print(y_test)
tree.score(X_test,y_test)
 
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
 
x['age'].fillna(x['age'].mean(), inplace=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
dt = DictVectorizer(sparse=False)
print(x_train.to_dict(orient="record"))# 按行，样本名字为键，列名也为键，[{"1":1,"2":2,"3":3}]
x_train = dt.fit_transform(x_train.to_dict(orient="record"))
x_test = dt.fit_transform(x_test.to_dict(orient="record"))
 
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dt_predict = dtc.predict(x_test)
print(dtc.score(x_test, y_test))
print(classification_report(y_test, dt_predict, target_names=["died", "survived"]))
 
rfc = RandomForestClassifier() 
rfc.fit(x_train, y_train)
rfc_y_predict = rfc.predict(x_test)
print(rfc.score(x_test, y_test))
print(classification_report(y_test, rfc_y_predict, target_names=["died", "survived"]))
