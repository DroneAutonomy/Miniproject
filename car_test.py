import keras
import matplotlib.pyplot as plt
import pickle
import numpy as np


"""Load the TEST data"""
pickle_in = open("x_test.pickle", "rb")
x_test_loaded = pickle.load(pickle_in)
x_test_loaded = np.array(x_test_loaded)
pickle_in.close()

pickle_in = open("class_names.pickle", "rb")
class_names = pickle.load(pickle_in)
pickle_in.close()

"""Load learning model"""
model = keras.models.load_model("car_classifier_model")

#x_test_loaded = keras.utils.normalize(x_test_loaded, axis=1)


"""Predict the car"""
predictions = model.predict(x_test_loaded)

"""Show 10 first results"""
for i in range(0,10):
    plt.imshow(x_test_loaded[i,:,:,0], cmap =  "gray")
    print(predictions[i])
    print(class_names[np.argmax(predictions[i])])
    plt.show()


