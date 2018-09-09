import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #importing necessary modules
def computeCost(X, y, theta):    #for coputing cost function j
    func = np.power(((X * theta.T) - y), 2) #not that x and y are all in matrices and are we are using numpy to perform math operationd
    return np.sum(func) / (2 * len(X)) #sum function is used for finding sum of all elements
def gradientDescent(X, y, theta, alpha, iters): #function for performing gradient descent

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






import os #os based operations in python
path = os.getcwd() + '\data\ex1data1.txt' #acquiring path if you want you can copy folder add and give it to path
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])#panda read function and storing in form of data

#print (data.head())#checking if data is acquired please cascade as comment after evaluation
#print (data.describe())#statisical basic function
#plt.show(data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8)))#ploting the graph of given data remove cascade to see
data.insert(0,'ones', 1)  #including 1s column in datframe
cols = data.shape[1] #1 returns a tuple measuring number of column
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols] #element selection by slicing the data frame
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))
#print(X.shape,y.shape,theta.shape)#checking shape of three converted matrix from dataframe cascade when not needed
#print(computeCost(X, y, theta))
alpha = 0.01  #initializing value such as learning rate and iteration
iters = 1000
g, cost = gradientDescent(X, y, theta, alpha, iters)
#print(g)
#print(computeCost(X, y, g))#this is way better than 32


#plotfunction for the graph
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'g', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

