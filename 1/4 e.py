import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def read_from_file(file):
    X = list()
    y = list()
    convert_y = {'red': -1, 'green': 1}
    for line in file.readlines()[1:]:
        input_data = line.rstrip('\n').split('\t')
        X.append(input_data[1:-1])
        y.append(convert_y[input_data[-1]])
    return np.array(X).astype(np.float64), np.array(y).astype(np.int64)


dataset = open("svmdata_e.txt", "r")
train_X, train_y = read_from_file(dataset)

testfile = open("svmdata_e_test.txt", "r")
test_X, test_y = read_from_file(testfile)

C = 1
for gamma in [1, 10, 100]:
    models = (
    svm.SVC(kernel='poly', degree=1, gamma=gamma, C=C),
    svm.SVC(kernel='poly', degree=2, gamma=gamma, C=C),
    svm.SVC(kernel='poly', degree=3, gamma=gamma, C=C),
    svm.SVC(kernel='poly', degree=4, gamma=gamma, C=C),
    svm.SVC(kernel='poly', degree=5, gamma=gamma, C=C),
    svm.SVC(kernel='poly', degree=6, gamma=gamma, C=C),
    svm.SVC(kernel="sigmoid", gamma=gamma, C=C),
    svm.SVC(kernel='rbf', gamma=gamma, C=C))

    models = (clf.fit(train_X, train_y) for clf in models)
    titles = (
    "SVC with poly kernel (degree=1)",
    "SVC with poly kernel (degree=2)",
    "SVC with poly kernel (degree=3)",
    "SVC with poly kernel (degree=4)",
    "SVC with poly kernel (degree=5)",
    "SVC with poly kernel (degree=6)",
    "SVC with sigmoid kernel",
    "SVC with rbf kernel",
    )

    fig, sub = plt.subplots(4, 2, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    X0, X1 = test_X[:, 0], test_X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.RdYlGn, alpha=0.8)
        ax.scatter(X0, X1, c=test_y, cmap=plt.cm.RdYlGn, s=20, edgecolors="y")
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel("Sepal length")
        ax.set_ylabel("Sepal width")
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.grid()
    plt.show()