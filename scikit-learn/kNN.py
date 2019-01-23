import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.externals import joblib

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

X_train,X_test,y_train,y_test = train_test_split(iris_X,iris_y,test_size=0.3)
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
joblib.dump(knn, 'save/knn.pkl')
knn.predict(X_test)
knn.score(X_test, y_test)

k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, iris_X, iris_y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
