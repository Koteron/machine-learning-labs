import csv
import random

import numpy as np
from sklearn.model_selection import train_test_split

classes = [-1, 1]
TRAIN_SIZE = 0.80

dataset = list()
with open('nn_1.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    reader.__next__()
    for row in reader:
        dataset.append((np.array(row[:-1]).astype(np.float64), int(row[-1])))

print(dataset)

train_data, test_data = train_test_split(dataset, train_size=TRAIN_SIZE)

# Input and output sizes
INPUT_DIM = 2
OUT_DIM = 2

# Input parameters
x = np.random.randn(INPUT_DIM)

# Trainable parameters
W = np.random.randn(INPUT_DIM, OUT_DIM)
b = np.random.randn(OUT_DIM)


def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)


def sparse_cross_entropy(network_output, right_value):
    return -np.log(network_output[0, right_value])  # -np.sum(y*np.log(z))


def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full


LEARNING_RATE = 0.001
EPOCH_NUM = 100
x = np.random.randn(1, INPUT_DIM)
loss_arr = list()
for epoch in range(EPOCH_NUM):
    random.shuffle(train_data)
    for i in range(len(train_data)):
        x, y = train_data[i]
        while (len(x) == 1):
            x = x[0]
        x.shape = (1, 2)
        y = 0 if y == -1 else 1

        # Forward propagation
        t = x @ W + b
        z = softmax(t)
        E = sparse_cross_entropy(z, y)

        # Backward propagation
        y_full = to_full(y, OUT_DIM)
        dE_dt = z - y_full

        dE_dW = x.T @ dE_dt
        dE_db = dE_dt

        # Update
        W = W - LEARNING_RATE * dE_dW
        b = b - LEARNING_RATE * dE_db

        loss_arr.append(E)


def predict(x):
    t = x @ W + b
    z = softmax(t)
    return z


def calc_accuracy():
    correct = 0
    for x, y in test_data:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct / len(test_data)
    return acc


print("Accuracy: " + str(calc_accuracy()))

import matplotlib.pyplot as plt

plt.plot(loss_arr)
plt.show()
