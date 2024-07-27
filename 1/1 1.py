from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = open("tic_tac_toe.txt", "r")
X = []
y = []
convert_X = {'x': 1, 'o': 0, 'b': 2}
convert_y = {'positive': 1, 'negative': -1}

for line in dataset:
    input_data = line.rstrip('\n').split(',')

    X.append([convert_X.get(x, -1) for x in input_data[:9]])
    y.append(convert_y.get(input_data[-1], -1))

dataset.close()

train_size = 0.02
GaussianNB_object = GaussianNB()
accuracy_score_of_y_prediction_test = []
accuracy_score_of_y_prediction_train = []
list_of_train_sizes = []

for i in range(0, 49):
    list_of_train_sizes.append(train_size)
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=train_size)
    GaussianNB_object.fit(train_X, train_y)
    y_prediction_test = GaussianNB_object.predict(test_X)
    accuracy_score_of_y_prediction_test.append(accuracy_score(test_y, y_prediction_test))
    print("Accuracy test:", accuracy_score_of_y_prediction_test[-1], "train_size=", train_size)
    y_prediction_train = GaussianNB_object.predict(train_X)
    accuracy_score_of_y_prediction_train.append(accuracy_score(train_y, y_prediction_train))
    print("Accuracy train:", accuracy_score_of_y_prediction_train[-1], "train_size=", train_size)
    print()
    train_size += 0.02

plt.plot(list_of_train_sizes, accuracy_score_of_y_prediction_test, label='Test accuracy')
plt.plot(list_of_train_sizes, accuracy_score_of_y_prediction_train, label='Train accuracy')
plt.xlabel("Train size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()