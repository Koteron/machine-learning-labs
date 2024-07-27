import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))
    return xx, yy


def read_from_file(file):
    X = list()
    y = list()
    convert_y = {'red': -1, 'green': 1}
    for line in file.readlines()[1:]:
        input_data = line.rstrip('\n').split('\t')
        X.append(input_data[1:-1])
        y.append(convert_y[input_data[-1]])
    return np.array(X).astype(np.float64), np.array(y).astype(np.int64)

def make_grafic(train_X, train_y, test_X, test_y, c, name):
    clf1 = svm.SVC(kernel='linear', C=c)
    clf1.fit(train_X, train_y)
    X = test_X
    y = test_y
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    Z = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn)
    plt.scatter(X0, X1, c=y, cmap=plt.cm.RdYlGn, s=20, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('SVC with linear kernel ' + name)
    plt.show()

dataset = open("svmdata_b.txt", "r")
train_X, train_y = read_from_file(dataset)

testfile = open("svmdata_b_test.txt", "r")
test_X, test_y = read_from_file(testfile)


c = 0.1
c_test = list()
c_train = list()
c_optimal = list()
for i in range(10000):
    clf1 = svm.SVC(kernel='linear', C=c)
    clf1.fit(train_X, train_y)
    clf_predictions1 = clf1.predict(test_X)
    if accuracy_score(test_y, clf_predictions1) == 1.0:
        c_test.append(c)
    clf_predictions2 = clf1.predict(train_X)
    if accuracy_score(train_y, clf_predictions2) == 1.0:
        c_train.append(c)
    c += 0.1

print(c_train[0])
print(c_test[0])

make_grafic(train_X, train_y, train_X, train_y, c_train[0], 'train')
make_grafic(train_X, train_y, test_X, test_y, c_train[0], 'test')
make_grafic(train_X, train_y, train_X, train_y, c_test[0], 'train')
make_grafic(train_X, train_y, test_X, test_y, c_test[0], 'test')