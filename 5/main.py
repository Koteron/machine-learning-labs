import sklearn
import prettytable
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge


def run1():
        def print_accuracy_test(testing_data, classifier):
            table = prettytable.PrettyTable(['var', 'score'])
            for foxchunk in testing_data:
                X_train, X_test, y_train, y_test = train_test_split(foxchunk[1], foxchunk[2].values.ravel(), test_size=0.25)
                classifier.fit(X_train, y_train)
                pred = classifier.predict(X_test)
                score = classifier.score(X_test, y_test)
                table.add_row([foxchunk[0], str(score)])
                # print(foxchunk[0] + " -> " + str(score))
            print(table)

        data = read_csv('reglab1.txt', delimiter='\t')
        z_data = pandas.DataFrame(data.loc[:, "z"])
        x_data = pandas.DataFrame(data.loc[:, "x"])
        y_data = pandas.DataFrame(data.loc[:, "y"])
        classifiers = [
            [LinearRegression(), "LinearRegression"],
            [KNeighborsRegressor(), "KNeighborsRegressor"],
            [DecisionTreeRegressor(), "DecisionTreeRegressor"],
            [BayesianRidge(), "BayesianRidge"]
        ]
        tett = pandas.concat([y_data, z_data], ignore_index=False,
sort=False,
                             axis=1)
        print(tett)
        data_test = [
            ["x(y)", y_data, x_data],
            ["x(z)", z_data, x_data],
            ["y(x)", x_data, y_data],
            ["y(z)", z_data, y_data],
            ["z(x)", x_data, z_data],
            ["z(y)", y_data, z_data],
            ["x(y, z)", pandas.concat([y_data, z_data], ignore_index=True,
                                      sort=False, axis=1), x_data],
            ["y(x, z)", pandas.concat([x_data, z_data], ignore_index=True,
                                      sort=False, axis=1), y_data],
            ["z(x, y)", pandas.concat([x_data, y_data], ignore_index=True,
                                      sort=False, axis=1), z_data]
        ]
        for classf in classifiers:
            print(classf[1])
            print_accuracy_test(data_test, classf[0])


def run2():
    data = read_csv('reglab.txt', delimiter='\t')
    data_y = pandas.DataFrame(data.loc[:, "y"])
    data_x1 = pandas.DataFrame(data.loc[:, "x1"])
    data_x2 = pandas.DataFrame(data.loc[:, "x2"])
    data_x3 = pandas.DataFrame(data.loc[:, "x3"])
    data_x4 = pandas.DataFrame(data.loc[:, "x4"])
    classifier = LinearRegression()
    data_test = [
        ["x1", data_x1],
        ["x2", data_x2],
        ["x3", data_x3],
        ["x4", data_x4],
        ["x1, x2", pandas.concat([data_x1, data_x2], ignore_index=True,
                                 sort=False, axis=1)],
        ["x1, x3", pandas.concat([data_x1, data_x3], ignore_index=True,
                                 sort=False, axis=1)],
        ["x1, x4", pandas.concat([data_x1, data_x4], ignore_index=True,
                                 sort=False, axis=1)],
        ["x2, x3", pandas.concat([data_x2, data_x3], ignore_index=True,
                                 sort=False, axis=1)],
        ["x2, x4", pandas.concat([data_x2, data_x4], ignore_index=True,
                                 sort=False, axis=1)],
        ["x3, x4", pandas.concat([data_x3, data_x4], ignore_index=True,
                                 sort=False, axis=1)],
        ["x1, x2, x3", pandas.concat([data_x1, data_x2, data_x3],
                                     ignore_index=True, sort=False, axis=1)],
        ["x1, x2, x4", pandas.concat([data_x1, data_x2, data_x4],
                                     ignore_index=True, sort=False, axis=1)],
        ["x1, x3, x4", pandas.concat([data_x1, data_x3, data_x4],
                                     ignore_index=True, sort=False, axis=1)],
    ]

    table = prettytable.PrettyTable(['var', 'RSS'])
    for foxchunk in data_test:
        X_train, X_test, y_train, y_test = train_test_split(foxchunk[1],
                                                  data_y, test_size=0.25)
        classifier.fit(X_train, y_train)
        pred = classifier.predict(X_test)
        table.add_row([foxchunk[0],
sklearn.metrics.mean_squared_error(y_test, pred)])
        print(table.get_string(sortby="RSS", reversesort=False))



def run3():
    data = read_csv('cygage.txt', delimiter='\t')
    X_without = pandas.DataFrame(data.loc[:, "Depth"])
    X_with = data.drop("calAge", axis=1)
    Y = pandas.DataFrame(data.loc[:, "calAge"])
    # no depth
    X_train, X_test, y_train, y_test = train_test_split(X_without, Y,
                                                        test_size=0.25)
    classifier = LinearRegression()
    classifier.fit(X_train, y_train)
    reg_without = classifier.predict([X_without.values[0], X_without.values[-
    1]])
    print("Without depth", classifier.score(X_test, y_test))
    # with depth
    X_train, X_test, y_train, y_test = train_test_split(X_with, Y,
                                                        test_size=0.25)
    classifier = LinearRegression()
    classifier.fit(X_train, y_train)
    reg_with = classifier.predict([X_with.values[0], X_with.values[-1]])
    print("With depth", classifier.score(X_test, y_test))
    # draw
    fig, ax = plt.subplots()
    ax.scatter(data.loc[:, "Depth"], data.loc[:, "calAge"], c='blue')
    ax.plot([0, 700], reg_without, c='r', label='Without depth')
    ax.plot([0, 700], reg_with, c='g', label='With depth')
    ax.legend()
    plt.show()


def run4():
    f = read_csv('longley.csv')
    f = f.drop('Population', axis=1)
    f = f.sample(frac=1)
    y = f.Employed.values
    x = f.drop('Employed', axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5,
                                                        shuffle=True)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    score = regr.score(X_test, y_test)
    print(f'Точность линейной регрессии: {score}')
    alphas = [10 ** (-3 + 0.2 * i) for i in range(26)]
    scores_test = []
    scores_train = []
    for alpha in alphas:
        rcv = Ridge(alpha=alpha)
        rcv.fit(X_train, y_train)
        scores_test.append(rcv.score(X_test, y_test))
        scores_train.append(rcv.score(X_train, y_train))

    print(f'Точность гребневой регрессии: {np.max(scores_test)}')
    plt.plot(alphas, scores_test, 'b')
    plt.plot(alphas, scores_train, 'orange')
    plt.legend(['Test data', 'Train data'])
    plt.show()


def run5():
    data = read_csv('eustock.csv', delimiter=',').to_numpy()
    titles = ['DAX', 'SMI', 'CAC', 'FTSE']
    columns = []
    for i in range(len(data[0])):
        columns.append(data[:, i])
    columns = np.array(columns)
    xticks = [i for i in range(1, len(columns[0]) + 1)]
    reals = []
    for column in columns:
        real, = plt.plot(xticks, column)
        reals.append(real)

    plt.legend(reals, titles)
    plt.grid(True)
    plt.xticks(())
    plt.show()
    xticks = np.arange(1, len(columns[0]) + 1)
    for column, title in zip(columns, titles):
        clf = LinearRegression()
        column = column.reshape(-1)
        xticks_reshaped = xticks.reshape(-1, 1)
        clf.fit(xticks_reshaped, column)
        pred = clf.predict(xticks_reshaped)
        plt.plot(xticks_reshaped, pred)

    plt.legend(titles)
    plt.grid(True)
    plt.xticks(())
    plt.show()

    print(data)
    data_avg = list()
    for row in data:
        data_avg.append((row[0]+row[1]+row[2]+row[3])/4)
    data_avg = np.array(data_avg)
    xticks = np.arange(1, len(columns[0]) + 1)
    clf = LinearRegression()
    data_avg = data_avg.reshape(-1)
    xticks_reshaped = xticks.reshape(-1, 1)
    clf.fit(xticks_reshaped, data_avg)
    pred = clf.predict(xticks_reshaped)
    plt.plot(xticks_reshaped, pred)

    plt.legend(titles)
    plt.grid(True)
    plt.xticks(())
    plt.show()
    #for row in data:




def run6():
    data = read_csv('JohnsonJohnson.csv', delimiter=',').to_numpy()
    qs = [[] for _ in range(4)]
    for i in data:
        qnum = int(i[0][-1])
        qs[qnum - 1].append(i)
    for i in range(len(qs)):
        qs[i] = np.array(qs[i])[:, 1].reshape(-1, 1)

    qs = np.array(qs)
    x_axis = range(len(qs[0]))
    years = np.arange(1960, 1981)
    all_years = np.sum(np.concatenate(qs, axis=1), axis=1).reshape(-1, 1)
    plt.figure(figsize=(20, 10))
    for q in qs:
        plt.plot(x_axis, q)

    #plt.plot(x_axis, all_years)
    plt.xticks(x_axis, years)
    plt.legend(('Q1', 'Q2', 'Q3', 'Q4'))
    plt.grid(True)
    plt.show()
    all_years = np.sum(np.concatenate(qs, axis=1), axis=1).reshape(-1, 1)
    plt.figure(figsize=(20, 10))
    preds_2016 = []
    clf = LinearRegression()
    yreshaped = years.reshape(-1, 1)
    color = ['r', 'g', 'b', 'y']
    for q, c in zip(qs, color):
        clf.fit(yreshaped, q.reshape(-1))
        pred = clf.predict(yreshaped)
        plt.plot(years, pred, c=c, ls='--')
        plt.plot(years, q, c=c)
        preds_2016.append(clf.predict([[2016]])[0])

    plt.xticks(years, [str(i) for i in years])
    plt.legend(('Q1', 'Q1r', 'Q2', 'Q2r', 'Q3', 'Q3r', 'Q4', 'Q4r'))
    plt.grid(True)
    plt.show()
    for p, i in zip(preds_2016, range(1, 5)):
        print('Q' + str(i), p, sep='\t')

    clf.fit(yreshaped, all_years)
    print(clf.predict([[2016]])[0])


def run7():
    def regress(x_data, y_data):
        train_x, test_x, train_y, test_y = train_test_split(x_data, y_data,
train_size=0.8)
        regression = LinearRegression()
        regression.fit(train_x, train_y)
        x = [[10], [15], [20], [25], [30], [35], [40], [45], [50]]
        pred = regression.predict(x)
        tt = [i for i in range(10, 55, 5)]
        print(tt)
        plt.plot(tt, pred)
        plt.xlabel('Speed')
        plt.ylabel('Braking distance')
        plt.show()

    path = "cars.csv"
    X = []
    Y = []
    file = open(path, "r")
    for row in file.readlines()[1:]:
        arr = row.rstrip("\n").replace('"', "").split(",")
        X.append(int(arr[0]))
        Y.append(int(arr[1]))
    file.close()
    X = np.array(X)
    Y = np.array(Y)
    plt.scatter(X, Y, c='blue')
    # plt.grid(True)
    plt.xlabel("Speed")
    plt.ylabel("Braking distance")
    plt.show()
    regress(X.reshape(-1, 1), Y)


def run8():
    data = read_csv('svmdata6.txt', delimiter='\t').to_numpy()
    data
    X = data[:, 0].reshape(-1, 1)
    Y = data[:, 1]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
    clf = SVR(C=1, kernel='rbf')
    epsilons = np.arange(0, 2, 0.1)
    errors = []
    for e in epsilons:
        clf.epsilon = e
        clf.fit(x_train, y_train)
        err = mean_squared_error(y_test, clf.predict(x_test))
        errors.append(err)

    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, errors)
    plt.xlabel('Epsilon')
    plt.ylabel('MSE')
    plt.show()

def run9():
    data = read_csv('nsw74psid1.csv', delimiter=',').to_numpy()
    Y = data[:, -1]
    X = np.delete(data, [len(data[0]) - 1], axis=1)
    print(X)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
    classifiers = [
        DecisionTreeRegressor(),
        LinearRegression(),
        SVR()
    ]
    for clf in classifiers:
        clf.fit(x_train, y_train)
        print('{:>22}: {:f}'.format(clf.__class__.__name__, clf.score(x_test,
                                                                  y_test)))
if __name__ == '__main__':
    #run1()
    #run2()
    #run3()
    #run4()
    #run5()
    #run6()
    #run7()
    #run8()
    run9()
