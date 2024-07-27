from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = list()
y = list()
convert_y = {'spam': 0, 'nonspam': -1}
with open('glass.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    reader.__next__()
    for row in reader:
        X.append(np.array(row[1:-1]).astype(np.float64))
        y.append(np.array(row[-1]).astype(np.int64))

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.70)
classes = np.zeros(7).astype(np.int64)
accuracy_score_of_y_prediction_test = list()
accuracy_score_of_y_prediction_train = list()
count_neighbors = list()

for neighbors in range(1, 16):
    count_neighbors.append(neighbors)
    KNeighborsClassifier_object = KNeighborsClassifier(n_neighbors=neighbors)
    KNeighborsClassifier_object.fit(train_X, train_y)
    y_prediction_test = KNeighborsClassifier_object.predict(test_X)
    accuracy_score_of_y_prediction_test.append(accuracy_score(test_y, y_prediction_test))
    y_prediction_train = KNeighborsClassifier_object.predict(train_X)
    accuracy_score_of_y_prediction_train.append(accuracy_score(train_y, y_prediction_train))
    prediction = KNeighborsClassifier_object.predict([[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]])
    classes[prediction[0] - 1] += 1
    print(classes)


plt.plot(count_neighbors, accuracy_score_of_y_prediction_test, label='Test accuracy')
plt.plot(count_neighbors, accuracy_score_of_y_prediction_train, label='Train accuracy')
plt.xlabel("Count neighbors")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


classes = np.zeros(7).astype(np.int64)
for metric in ['euclidean', 'manhattan', 'chebyshev', 'minkowski']:
    KNeighborsClassifier_object = KNeighborsClassifier(metric=metric)
    KNeighborsClassifier_object.fit(train_X, train_y)
    y_prediction_test = KNeighborsClassifier_object.predict(test_X)
    print(accuracy_score(test_y, y_prediction_test))
    prediction = KNeighborsClassifier_object.predict([[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]])
    classes[prediction[0] - 1] += 1

print(classes)