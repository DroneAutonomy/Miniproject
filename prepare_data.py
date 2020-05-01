import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import cv2
import sys
import math
import h5py
import pickle


TRAIN_DATA_DIR = "cars_train"
TEST_DATA_DIR = "cars_test"
IMG_SIZE = 100
training_data = []





"""Getting class identifiers (labels) from the matlab file"""
class_values_mat = scipy.io.loadmat("cars_train_annos.mat")
class_names_mat = scipy.io.loadmat("cars_meta.mat")

class_values_data = [[img_nr.flat[0] for img_nr in item] for item in class_values_mat["annotations"][0]] # data into list
class_names_list = [item.flat[0] for item in class_names_mat["class_names"][0]]

print(class_values_data)
"""Appending the only data we want: labels"""
train_label_list = []
for i in class_values_data :
    train_label_list.append([i[4]])



"""Iterating through all images, resizing them, appending them to a list"""
counter= 0
for img in sorted(os.listdir(TRAIN_DATA_DIR)):
    try:
        """Converting images to grayscale to reduce data size"""
        img_array = cv2.imread(os.path.join(TRAIN_DATA_DIR, img), cv2.IMREAD_GRAYSCALE)
        """Resizing all images into same size"""
        resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        """Inserting all training data into an array"""

        training_data.append([resized_array, train_label_list[counter]])
        counter += 1
        sys.stdout.write('\r' + "Loading: " + str(counter/8144*100) + "%")
    except Exception as e:
        print(e)



plt.imshow(training_data[0][0], cmap="gray")
print("\n" + class_names_list[training_data[0][1][0]])

print(len(training_data))


x_train = []
y_train = []

for features, label in training_data:
    x_train.append(features)
    y_train.append(label)

"""In order to use the data it must be in a numpy array - keras uses numpy array as input"""
x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE,1)


"""Save the data, so it does not have to load each time"""
pickle_out = open("x_train.pickle", "wb")
pickle.dump(x_train, pickle_out)
pickle_out.close()
pickle_out = open("y_train.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()


plt.show()