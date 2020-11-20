# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 07:50:10 2020

@author: Shithil. S. Shetty
"""
"""Task:Predict the percentage of an student based on the no. of study hours."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values

"""Splitting dataset into training set and test set"""
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.50, random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor=regressor.fit(X_train,Y_train)

Y_pred=regressor.predict(X_test)

"""Visualising the Training set results"""
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Percentage vs No. of hours (Training set)')
plt.xlabel('No. of hours')
plt.ylabel('Percentage')
plt.show()

"""Visualising the Test set results"""
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Percentage vs No. of hours (Test set)')
plt.xlabel('No. of hours')
plt.ylabel('Percentage')
plt.show()


""" Predicted score if a student studies for 9.25 hrs/ day"""
x=([[9.25]])
y=regressor.predict(x)

"""Anwswer"""
print(y)

