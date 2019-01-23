import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
from sklearn.learning_curve import validation_curve
import matplotlib.pyplot as plt
from sklearn.externals import joblib

#make datasets
X, y = make_classification(
    n_samples=300, n_features=2,
    n_redundant=0, n_informative=2,
    random_state=22, n_clusters_per_class=1,
    scale=100)

#matplotlib make map
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

#normalization
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#kernel：linear，poly，rbf，sigmod
clf = SVC(kernel='rbf', C=1)
clf.fit(X_train, y_train)
predict = clf.predict(X_test)
score = clf.score(X_test, y_test)
print(predict)
print(score)
joblib.dump(clf, 'save/clf.pkl')

clf2 = SVC(kernel='rbf', C=1)
#cross validation
scores = cross_val_score(clf2, X, y, cv=5)
scores2 = scores.mean()
print(scores2)

param_range = np.logspace(-6, -2.3, 5)
#scoring：accuracy，Mean squared error
train_loss, test_loss = validation_curve(SVC(), X, y, param_name='gamma', param_range=param_range, cv=10, scoring='mean_squared_error')

train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

#matplotlib make map
plt.plot(param_range, train_loss_mean, 'o-', color="r", label="Training")
plt.plot(param_range, test_loss_mean, 'o-', color="g", label="Cross-validation")

plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
