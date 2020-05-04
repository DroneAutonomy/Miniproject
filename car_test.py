import tensorflow as tf
import keras as keras
import matplotlib.pyplot as plt
import numpy as np


"""Load the data"""
pickle_in = open("x_train.pickle", "rb")
x_train_loaded = np.pickle.load(pickle_in)
pickle_in.close()



tf.keras.models.load_model("model_name")
predictions = model.predict([x_test])