import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import cv2
import sys
import math

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


"""Name of the model"""
NAME = "Car-Classifier-cnn-64x2-{}".format(int(time.time()))

"""Log to TensorBoard"""
tensorboard = TensorBoard(log_dir='/Users/lasseskjotthansen/Desktop/images/logs/{}'.format(NAME))

"""Load the data"""
pickle_in = open("x_train.pickle", "rb")
x_train_loaded = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("y_train.pickle", "rb")
y_train_loaded = pickle.load(pickle_in)
y_train_loaded = np.array(y_train_loaded)
pickle_in.close()


x_train_loaded = tf.keras.utils.normalize(x_train_loaded, axis=1)


model = tf.keras.models.Sequential()
"""Start with the convolutional layer"""
model.add(tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=x_train_loaded.shape[1:]))

"""Adding maxpooling layer with size (2,2)"""
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

"""Doing the same one more time"""
model.add(tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128))

"""output layer """
model.add(tf.keras.layers.Dense(197, activation=tf.nn.softmax))


model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train_loaded, y_train_loaded, batch_size=32, validation_split=0.1, epochs=8, callbacks=[tensorboard])

