import keras
import matplotlib.pyplot as plt
import pickle
import numpy as np


"""Load the data"""
pickle_in = open("x_train.pickle", "rb")
x_train_loaded = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("y_train.pickle", "rb")
y_train_loaded = pickle.load(pickle_in)
y_train_loaded = np.array(y_train_loaded)
pickle_in.close()

pickle_in = open("class_names.pickle", "rb")
class_names = pickle.load(pickle_in)
pickle_in.close()



model = keras.models.load_model("car_classifier_model")


predictions = model.predict(x_train_loaded)

print(class_names[np.argmax(predictions[0])])


plt.imshow(x_train_loaded[0,:,:,0], cmap =  "gray")
plt.show()