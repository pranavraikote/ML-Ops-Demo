# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 02:18:36 2018

@author: Pranav
"""

from keras.models import Sequential
from keras.layers import Dense  
import numpy

#Set a random seed to get same output score (Useful for comparing results)
numpy.random.seed(7)

#Load the dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

#Split the data set
X = dataset[:,0:8]
Y = dataset[:,-1]

#Define the network
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train the network
model.fit(X, Y, epochs=10, batch_size=100)

#Test the network
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Predictions
predictions = model.predict(X)
rounded = [round(x[0]) for x in predictions]






