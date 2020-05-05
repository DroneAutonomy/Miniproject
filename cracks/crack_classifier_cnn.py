import pickle
import time
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.keras.Callbacks import TensorBoard


"""Name of the model"""
#NAME = "Car-Classifier-cnn-64x2-{}".format(int(time.time()))

"""Log to TensorBoard"""
#tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

"""Load the data"""
x_train_loaded = pickle.load(open("x_train.pickle", "rb"))
y_train_loaded = pickle.load(open("y_train.pickle", "rb"))


"""Normalizing data"""
x_train_loaded = keras.utils.normalize(x_train_loaded, axis=1)

"""Feedforward network"""
model = keras.models.Sequential()
"""Start with the convolutional layer"""
model.add(keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=x_train_loaded.shape[1:]))
"""Adding maxpooling layer with size (2,2)"""
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

"""Doing the same one more time"""
model.add(keras.layers.Conv2D(64, (3,3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation="relu"))

"""output layer """
model.add(keras.layers.Dense(1, activation="sigmoid"))

"""Compiling model, defining loss function, optimizer and metrics"""
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train_loaded, y_train_loaded, batch_size=32, validation_split=0.2, epochs=2)

model.save("crack_identifier_model")

