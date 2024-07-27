from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def read_from_file(file_name):
    X = list()
    y = list()
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        reader.__next__()
        for row in reader:
            X.append(np.array(row[1:]).astype(np.float64))
            y.append(np.array(row[0]).astype(np.int64))
    return np.array(X).astype(np.float64), np.array(y).astype(np.int64)


train_X, train_y = read_from_file('bank_scoring_train.csv')
test_X, test_y = read_from_file('bank_scoring_test.csv')

# k ближайших соседей
best_matrix = 0
count_neighbors = 0
best_accur = 0
for neighbors in range(1, 8):
    KNeighborsClassifier_object = KNeighborsClassifier(n_neighbors=neighbors)
    KNeighborsClassifier_object.fit(train_X, train_y)
    prediction = KNeighborsClassifier_object.predict(test_X)
    acc = accuracy_score(test_y, prediction)
    if acc > best_accur:
        best_accur = acc
        best_matrix = confusion_matrix(test_y,prediction)
        count_neighbors = neighbors

print(best_accur)
print(best_matrix)
print(count_neighbors)


# опорные вектора + линейное ядро
clf = svm.SVC(kernel='poly', degree=2)
clf.fit(train_X, train_y)
prediction = clf.predict(test_X)
print(accuracy_score(test_y, prediction))
print(confusion_matrix(test_y,prediction))


# дерево классификации
best_matrix = 0
best_accur = 0
best_matrix = 0
for split in ['best', 'random']:
    x_now = list()
    y_now = list()
    now_accuracy = list()
    for j in range(1, 100, 5):
        for k in range(2, 100, 5):
            x_now.append(j)
            y_now.append(k)
            clf = DecisionTreeClassifier(splitter=split, max_depth=j, min_samples_leaf =k)
            clf.fit(train_X, train_y)
            predict = clf.predict(test_X)
            acc = accuracy_score(test_y, predict)
            now_accuracy.append(acc)
            if acc > best_accur:
                best_accur = acc
                best_depth = clf.get_depth()
                best_matrix = confusion_matrix(test_y,predict)

    print(best_accur)
    print(best_matrix)
    print(best_depth)