import pydotplus
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import csv
import numpy as np
from six import StringIO
from sklearn.tree import DecisionTreeClassifier


import os
os.environ["PATH"] += os.pathsep + 'C:/Users/vikto/anaconda3/Library/bin/Graphviz-10.0.1-win64/bin'
X = list()
x_now = list()
convert_y = {'y': 1, 'n': 0}
with open('spam7.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    reader.__next__()
    for row in reader:
        X.append(np.array(row[0:-1]).astype(np.float64))
        x_now.append(np.array(convert_y[row[-1]]).astype(np.int64))

train_X, test_X, train_y, test_y = train_test_split(X, x_now, train_size=0.75)
clf = DecisionTreeClassifier()
clf.fit(train_X, train_y)
prediction = clf.predict(test_X)
print(confusion_matrix(test_y,prediction))
print(accuracy_score(test_y, clf.predict(test_X)))
print(clf.get_depth())
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("predict.pdf")
graph.write_jpg("predict.jpg")

x = list()
y = list()
accuracy = list()
i = 0
max_accur = 0
max_depth = 0
depth = 0
max_min_samples_leaf = 0
confusion_matrixes = list()
for split in ['best', 'random']:
    x_now = list()
    y_now = list()
    now_accuracy = list();
    for j in range(1, 100, 5):
        for k in range(2, 100, 5):
            x_now.append(j)
            y_now.append(k)
            clf = DecisionTreeClassifier(splitter=split, max_depth=j, min_samples_leaf =k)
            clf.fit(train_X, train_y)
            predict = clf.predict(test_X);
            acc = accuracy_score(test_y, predict)
            now_accuracy.append(acc)
            if acc > max_accur:
                max_accur = acc
                depth = clf.get_depth()
                max_depth = j
                max_min_samples_leaf = k
                dot_data = StringIO()
                tree.export_graphviz(clf, out_file=dot_data)
                graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                graph.write_jpg("predict_best.jpg")
                confusion_matrixes.append(confusion_matrix(test_y,predict))
    x.append(x_now)
    y.append(y_now)
    accuracy.append(now_accuracy)


print("Accuracy:", max_accur)
print("Depth:", depth)
print("Max depth:", max_depth)
print("Max min samples leaf:", max_min_samples_leaf)
print(confusion_matrixes[-1])
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x[0], y[0], accuracy[0], label='Best splitter')
ax.scatter(x[1], y[1], accuracy[1], label='Random splitter')
plt.xlabel("Max depth")
plt.ylabel("Min samples leaf")
ax.grid()
ax.legend()
plt.show()