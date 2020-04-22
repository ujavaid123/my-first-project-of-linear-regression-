# my-first-project-of-linear-regression-
Linear Regression on Diabetes Dataset 
# Import some Libraries 

import pandas as pd
import numpy as np
import csv
import pandas
from sklearn import dataset

# Load the Dataset

from sklearn.datasets import load_diabetes

diabetes_data = load_diabetes()

df = pd.DataFrame(data=diabetes_data['data'], columns=diabetes_data['feature_names'])
df['target'] = diabetes_data['target']

print(df.shape)

print(df.head())

# defining our x and y variables
x= df.iloc[:,:-1].values
y= df.iloc[:,-1].values
m ,n = x.shape
print('x[0]={},y[0]={}'.format(x[0],y[0]))

#Appending column of ones to make it for intercept term
z= np.ones(m)
z= z.reshape(m,1)
x = np.append(z,x , axis =1)
print('x[0]={} ,y[0]={}'.format(x[0],y[0]))

# Declaring values of Theta
m,n = x.shape
theta = np.zeros(n)
theta = theta.reshape(n,1)
y = y.reshape(-1,1)


#computing the cost
def computeCost(x, y, theta):
    temp = np.dot(x, theta) - y
    return np.sum(np.power(temp, 2)) / (2*m)
J = computeCost(x, y, theta)
print(J)


# Finding the optimal parameters using Gradient Descent

iterations = 50000
alpha = 0.01
def gradientDescent(x, y, theta, alpha, iterations):
    for _ in range(iterations):
        temp = np.dot(x, theta) - y
        temp = np.dot(x.T, temp)
        theta = theta - (alpha / m) * temp
    return theta
theta = gradientDescent(x, y, theta, alpha, iterations)
print(theta)

# We now have optimized values of Thetas (intercept and slope ). so we can find minimum cost

J = computeCost(x, y, theta)
print(J)
