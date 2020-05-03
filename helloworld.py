import tensorflow as tf
import keras as keras
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist  #images of hand-written images 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""Normalize data:"""
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
"""Building the model:"""
model = tf.keras.models.Sequential() # using feedforward

"""Input layer: We want to flatten the data, so it becomes a single dimensional array"""
model.add(tf.keras.layers.Flatten())

"""2 Hidden Layers: 128 neurons and activation funciton: Rectified Linear (RELU)"""
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))


"""Output layer: Softmax function - gives probability distribution where total is 1A"""
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))


"""Compile model: defining the algorithms used to train the model"""
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=["accuracy"])

"""Training the model"""
model.fit(x_train, y_train, epochs=3)


"""Calculating validation loss and accuracy"""

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

plt.imshow(x_train[0], cmap =  plt.cm.binary)
plt.show()

