#Author(Shyam Srinivasan)
import numpy as np
import matplotlib.pyplot as plt #importing numpy and matplot libraries
#Function for ploting standard un normalised DATA x,y
def plotData(X,y):
#classifying datas based on 1s and 0s    
    posx = []
    posy = []
    negx = []
    negy = []
    for i in range(len(y)):
        if y[i] == 1:
            posx.append(X[i][0])
            posy.append(X[i][1])
        else:
            negx.append(X[i][0])
            negy.append(X[i][1])
#ploting Data x,y from accepted data 
    plt.plot(posx,posy,'k*',label='Admitted')
    plt.plot(negx,negy,'yo',label='Not admitted')
    plt.ylabel('Exam 2 score')
    plt.xlabel('Exam 1 score')
    plt.legend()

#standard sigmoid function used for logistic regresion
def sigmoid(z):
    return 1/(1+np.exp(-z))
#function used for calculating cost
def cost(theta,X,y):
    XbyT = X*theta
    return (-y.T*np.log(sigmoid(XbyT))-(1-y).T*np.log(1-sigmoid(XbyT)))/y.shape[0]
#gradient function  please note this is a portion of the formula which would be used in gdbt function
def gradient(theta,X,y):
    return X.T*(sigmoid(X*theta)-y)/y.shape[0]
#To perform the gradient descent using logistic regression    
def gdbt(theta,X,y,alpha,beta,iter,tol):
    
    
    
    # Input:
        # theta: Initial value
        # X: Training data (input)
        # y: Training data (output)
        # alpha: Parameter for line search, denoting the cost function will be descreased by 100xalpha percent
        # beta: Parameter for line search, denoting the "step length" t will be multiplied by beta
        # iter: Maximum number of iterations
        # tol: The procedure will break if the square of the 2-norm of the gradient is less than the threshold tol
        #Advanced minimisation based on truncated newton method used.
    for i in range(iter):
        grad = gradient(theta,X,y)
        delta = -grad
        if grad.T*grad < tol:
            print 'Terminated due to stopping condition with iteration number',i
            return theta
        J = cost(theta,X,y)
        alphaGradDelta = alpha*grad.T*delta
        # begin line search
        t = 1
        while cost(theta+t*delta,X,y) > J+t*alphaGradDelta:
            t = beta*t
        # end line search

        # update
        theta = theta+t*delta  #minimised theta
    return theta
    
def featureNormalize(X): # feature normalisation function when the ranges between the data is very huge
    mu = np.mean(X,axis=0)
    
    sigma = np.std(X,axis=0)
    X_norm = np.divide(X-mu,sigma)
    return (X_norm,mu,sigma)
#stadard predict function using theta and x please note h in logistic regression uses sigmoid func 
def predict(theta,X):
    return np.matrix([1 if sigmoid(x*theta)>=0.5 else 0 for x in X]).T
    
# Accepting the Data
f = open('ex2data1.txt')
X = []
y = []
for line in f:
    data = line.split(',')
    X.append([float(data[0]),float(data[1])])
    y.append(float(data[2]))
#plotting Data
plotData(X,y)
plt.show()
myx = np.arange(-10,10,.1)#testing sigmoid function
plt.plot(myx,sigmoid(myx))
plt.show()
plotData(X,y)
#Matrix conversion
X = np.matrix(X)
y = np.matrix(y).T
#feature normalisatio done using normalisation function
normalized = featureNormalize(X)
X_norm = normalized[0]
mu = normalized[1]
sigma = normalized[2]


#  Compute cost and gradient
X = np.c_[np.ones(X.shape[0]),X]
X_norm = np.c_[np.ones(X_norm.shape[0]),X_norm]
initial_theta = np.matrix(np.zeros(X.shape[1])).T

costZero = cost(initial_theta,X,y)
gradientZero = gradient(initial_theta,X,y)
#initial cost and gradient at initial value for comparison sake
print 'Cost at initial theta (zeros):',costZero
print 'Gradient at initial theta (zeros):'
print gradientZero

# learning rate initialisation and other optimisation constant 
alpha = 0.01
beta = 0.8

thetaUnnorm = gdbt(initial_theta,X_norm,y,alpha,beta,1000,1e-8)
# Note above one uses normalized data, To get the unnormalized theta below code can be used please not directly using unnormalized data cause large penalty in terms of efficiencey
theta = thetaUnnorm

theta[0] -= sum(np.multiply(thetaUnnorm[1:],mu.T)/sigma.T)
for i in range(1,len(theta)):
    theta[i] = thetaUnnorm[i]/sigma[0,i-1]
#matrix containting all prediction with our derived theta
p = predict(theta,X)
print 'Cost at theta found by gdbt:',cost(theta,X,y)
print 'theta:'
#printing values of theta
print theta
#ploting the h intercept
score1 = np.squeeze(np.asarray(np.matrix(np.linspace(30,100,num=200))))
score2 = np.squeeze(np.asarray((-theta[0]-np.multiply(score1,theta[1]))/theta[2]))

plt.plot(score1,score2)

print 'For a student with scores 45 and 85, we predict an admission',sigmoid(np.matrix([1,45,85])*theta)

print 'Train accuracy:',len(y)-np.sum(np.abs(y-p))
#show the final plot with h intercept along with data
plt.show()


