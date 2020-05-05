import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import cv2
import sys
import math
import h5py
import pickle


TRAIN_DATA_DIR = "concrete_cracks_images"
TEST_DATA_DIR = "concrete_cracks_test_images"
IMG_SIZE = 100
training_data_negative = []
training_data_positive = []
testing_data_positive = []
testing_data_negative = []
positive = 1
negative = 0

def import_image_data (dir, label, folder_size, list):
    counter = 0
    for img in sorted(os.listdir(dir)):
        try:
            """Converting images to grayscale to reduce data size"""
            img_array = cv2.imread(os.path.join(dir, img), cv2.IMREAD_GRAYSCALE)

            """Resizing all images into same size"""
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

            """Inserting all training data into an array"""
            list.append([resized_array, label])
            counter += 1
            sys.stdout.write('\r' + "Loading: " + str(counter / folder_size * 100) + "%")
        except Exception as e:
            print(e)


"""Getting image data from images WITHOUT cracks"""
import_image_data(TRAIN_DATA_DIR + "/Negative", negative, 16000, training_data_negative)
"""Getting image data from images WITH cracks"""
import_image_data(TRAIN_DATA_DIR + "/Positive", positive, 16000, training_data_positive)

"""Combining data and shuffling it afterwards. Shuffling ensure that our model does not get 16000 positive
 followed by 16000 negative images, which could effect the results"""
combined_data = np.array(training_data_negative + training_data_positive)
np.random.shuffle(combined_data)

x_train = []
y_train = []

"""dividing the data into two lists"""
for features, label in combined_data:
    x_train.append(features)
    y_train.append(label)


"""Getting TEST data:"""
import_image_data(TEST_DATA_DIR + "/Negative", negative, 4000, testing_data_negative)
import_image_data(TEST_DATA_DIR + "/Positive", positive, 4000, testing_data_positive)
x_test = []
y_test = []
combined_test_data = np.array(testing_data_negative + testing_data_positive)
np.random.shuffle(combined_test_data)
for features, label in combined_test_data:
    x_test.append(features)
    y_test.append(label)

"""In order to use the data it must be in a numpy array - keras uses numpy array as input"""
x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array(y_train)
x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = np.array(y_test)


"""Save the data, so it does not have to load each time"""
pickle.dump(x_train,  open("x_train.pickle", "wb"))
pickle.dump(y_train, open("y_train.pickle", "wb"))
pickle.dump(y_test, open("y_test.pickle", "wb"))
pickle.dump(x_test, open("x_test.pickle", "wb"))

plt.show()

