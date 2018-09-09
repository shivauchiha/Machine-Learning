from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #import necessary modules
def gradientDescent(X, y, theta, alpha, iters):   #matrix operation therefore same function can be used for n variables
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost
def computeCost(X, y, theta):    #for coputing cost function j
    func = np.power(((X * theta.T) - y), 2) #not that x and y are all in matrices and are we are using numpy to perform math operationd
    return np.sum(func) / (2 * len(X)) #sum function is used for finding sum of all elements

import os
path = os.getcwd() + '\data\ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
#print(data2.head(80))#checking if the txt is successfully read into panda dataframe cascade once checked
data2 = (data2 - data2.mean()) / data2.std()  #normalisation to even the features effects
#print(data2.head())
# add ones column
data2.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))
alpha = 0.01  #initializing value such as learning rate and iteration
iters = 1000

# perform linear regression on the data set
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
#print(computeCost(X2, y2, g2)) #checking the effectiveness in reduction of error please cascade after checking
#plotting graph



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(data2.Size, data2.Bedrooms,data2.Price, label='Training Data')
ax.legend(loc=3)

ax.set_xlabel('Size')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')

plt.show()



#ploting learning curve
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
