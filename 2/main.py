import csv
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.80
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_data = []
y_data = []
with open('nn_1.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        x_data.append(np.array(row[:-1]).astype(np.float64))
        y_data.append(1 if int(row[-1]) == 1 else 0)

# class1_x = list()
# class1_y = list()
# class2_x = list()
# class2_y = list()
# for i in range(len(x_data)):
#     if y_data[i] == 0:
#         class1_x.append(x_data[i][0])
#         class1_y.append(x_data[i][1])
#     else:
#         class2_x.append(x_data[i][0])
#         class2_y.append(x_data[i][1])
# plt.scatter(class1_x, class1_y, label="Class \"-1\"")
# plt.scatter(class2_x, class2_y, label="Class \"1\"")
# plt.legend()
# plt.show()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=TRAIN_SIZE)



x_train = tf.convert_to_tensor(np.array(x_train), dtype=tf.float64)
y_train = tf.convert_to_tensor(np.array(y_train), dtype=tf.int32)
x_test = tf.convert_to_tensor(np.array(x_test), dtype=tf.float64)
y_test = tf.convert_to_tensor(np.array(y_test), dtype=tf.int32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(55, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=32, epochs=2000)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
predictions = np.argmax(model.predict(x_test), axis=1)
print(tf.math.confusion_matrix(y_test, predictions))