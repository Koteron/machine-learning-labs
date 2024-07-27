import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('vehicle.csv')
X = data.drop('Class', axis=1)
y = data['Class']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Список базовых классификаторов
classifiers = [DecisionTreeClassifier(), GaussianNB()]
classifiers_names = ['Decision Tree', 'Naive Bayes']

# Построение ансамблей и оценка качества
results = {}
for clf, name in zip(classifiers, classifiers_names):
    accuracies = []
    for n_estimators in range(1, 51):  # Перебор от 1 до 50 классификаторов в ансамбле
        boosting_clf = AdaBoostClassifier(base_estimator=clf, n_estimators=n_estimators, random_state=42)
        boosting_clf.fit(X_train, y_train)
        y_pred = boosting_clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    results[name] = accuracies

# Визуализация результатов
plt.figure(figsize=(10, 8))
for name, accuracies in results.items():
    plt.plot(range(1, 51), accuracies, label=name)
plt.xlabel('Number of Classifiers in Ensemble')
plt.ylabel('Accuracy')
plt.title('Effect of Number of Classifiers on Boosting Performance')
plt.legend()
plt.show()