import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X, y)
x = np.array(X[:, 1].A1)
f = model.predict(X).flatten()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()