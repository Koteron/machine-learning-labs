import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('glass.csv')
X = data.drop(['Id', 'Type'], axis=1)
y = data['Type']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Список базовых классификаторов
classifiers = [DecisionTreeClassifier(), KNeighborsClassifier(), GaussianNB(), SVC()]
classifiers_names = ['Decision Tree', 'K-NN', 'Naive Bayes', 'SVC']

# Построение ансамблей и оценка качества
results = {}
for clf, name in zip(classifiers, classifiers_names):
    accuracies = []
    for n_estimators in range(1, 51):  # Перебор от 1 до 50 классификаторов в ансамбле
        bagging_clf = BaggingClassifier(base_estimator=clf, n_estimators=n_estimators, random_state=42)
        bagging_clf.fit(X_train, y_train)
        y_pred = bagging_clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    results[name] = accuracies

# Визуализация результатов
plt.figure(figsize=(10, 8))
for name, accuracies in results.items():
    plt.plot(range(1, 51), accuracies, label=name)
plt.xlabel('Number of Classifiers in Ensemble')
plt.ylabel('Accuracy')
plt.title('Effect of Number of Classifiers on Ensemble Performance')
plt.legend()
plt.show()