import pickle
import numpy as np
import keras
import matplotlib.pyplot as plt

"""Load the data"""
x_test_loaded = pickle.load(open("x_test.pickle", "rb"))
y_test_loaded = pickle.load(open("y_test.pickle", "rb"))

"""Normalising image data"""
x_test_loaded = keras.utils.normalize(x_test_loaded, axis=1)


model = keras.models.load_model("crack_identifier_model")



evaluate = model.evaluate(x_test_loaded, y_test_loaded, batch_size=128)
print("test loss, test acc" + str(evaluate))

predictions = model.predict(x_test_loaded)

print(predictions.shape)
for i in range(0, len(predictions)):
    plt.imshow(x_test_loaded[i, :, :, 0], cmap =  "gray")
    print(predictions[i])
    plt.show()


