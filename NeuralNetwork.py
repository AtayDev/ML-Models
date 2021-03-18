import tensorflow as tf
#Keras: API of tensoflow to write less code
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Load data
data=keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']

train_images = train_images/255.0
test_images = test_images/255.0
#train_images and the rest are numpy arrays
#print(type(train_images))

#Before Creating a Model we need to figur out the Architecture of Our NN because: Model=Architecture+Parameters.

#Input Layer: Images are Matrices of 28*28 so the input layer well have a 28*28 neurons that represent all the pixels of the image.

#Hidden Layer: A number of Specified Neurons That we should figure out by our Own. In this example 128 Neurons is fine.In this layer we will use an activation of "relu" to keep on the positive results.

#Output Layer: This layer will have 9 Neurons where each Neuron describe a class(0-9). We will be using in this layer an activation of "Softmax" to have a distrubited Probablitiy between the classes.
#In this case the loss function is the cross-entropy softmax

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(150, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

#Optimize the loss function to find  the parameters
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#Train the model with the training images
model.fit(train_images,train_labels,epochs=5)
#Test the accuracy
test_loss, test_acc=model.evaluate(test_images,test_labels)
print(test_acc)

#Make predictions with the Model

predictions=model.predict(test_images)

for i in range(5):
    print("Prediction:", class_names[np.argmax(predictions[i])],"Reality:",class_names[test_labels[i]])

















