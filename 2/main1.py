import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.80
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=32, epochs=20)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
predictions = np.argmax(model.predict(x_test), axis=1)
print(tf.math.confusion_matrix(y_test, predictions))