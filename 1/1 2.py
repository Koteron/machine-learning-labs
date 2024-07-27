from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
import numpy as np

X = list()
y = list()
convert_y = {'spam': 0, 'nonspam': -1}
with open('spam.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    reader.__next__()
    for row in reader:
        X.append(np.array(row[1:-1]).astype(np.float64))
        y.append(convert_y[row[-1]])

train_size = 0.02
GaussianNB_object = GaussianNB()
accuracy_score_of_y_prediction_test = list()
accuracy_score_of_y_prediction_train = list()
list_of_train_sizes = list()
for i in range(0, 49):
    list_of_train_sizes.append(train_size)
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=train_size)
    GaussianNB_object.fit(train_X, train_y)
    y_prediction_test = GaussianNB_object.predict(test_X)
    accuracy_score_of_y_prediction_test.append(accuracy_score(test_y, y_prediction_test))
    print("Accuracy test:", accuracy_score_of_y_prediction_test[-1], "train_size=" + str(train_size))
    y_prediction_train = GaussianNB_object.predict(train_X)
    accuracy_score_of_y_prediction_train.append(accuracy_score(train_y, y_prediction_train))
    print("Accuracy train:", accuracy_score_of_y_prediction_train[-1], "train_size=" + str(train_size))
    print()
    train_size += 0.02

plt.plot(list_of_train_sizes, accuracy_score_of_y_prediction_test, label='Test accuracy')
plt.plot(list_of_train_sizes, accuracy_score_of_y_prediction_train, label='Train accuracy')
plt.xlabel("Train size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()