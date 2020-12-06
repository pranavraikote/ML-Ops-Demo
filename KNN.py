# -*- coding: utf-8 -*-
"""
Created on Sun Dec 6 23:53:32 2020

@author: Pranav
"""

#Import modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


#Load the iris dataset
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target

#Optional, to see the data and labels
#print(iris_data)
#print(iris_labels)

#Split the data into training and testing data in 70:30 ratio
X_train, X_test, Y_train, Y_test = train_test_split(iris_data, iris_labels, test_size = 0.25)

#Create a KNearestClassifier with value of K=5
classifier=KNeighborsClassifier(n_neighbors = 5)

#Fit the data and build the model
classifier.fit(X_train, Y_train)

#Predict using the test data
y_pred=classifier.predict(X_test)

#Print the Confusion Matrix
print('Confusion matrix is as follows')
print(confusion_matrix(y_test, y_pred))

#Print Precision and Recall
print('Accuracy Metrics')
print(classification_report(y_test, y_pred))