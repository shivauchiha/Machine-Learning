import numpy as np
import matplotlib.pyplot as plt

#importing numpy and matplot libraies
#function for ploting 
def plotData(X,y):
# seperation of data based on negative and postive outcomes
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
#plotting the data as given in arguments
    plt.plot(posx,posy,'k*',label='y=1')
    plt.plot(negx,negy,'yo',label='y=0')
    plt.ylabel('Microchip Test 2')
    plt.xlabel('Microchip Test 1')
    plt.legend()
#sigmoid function as used in logistic regression
def sigmoid(z):
    return 1/(1+np.exp(-z))
#function for feautre mappping the matrix be like [1 , x1 ,x2,x1^2,x1x2...]
def mapFeature(X1,X2,degree):
    out = np.matrix(np.ones(X1.shape[0])).T
    for i in range(1,degree+1):
        for j in range(i+1):
            out = np.c_[out,np.multiply(np.power(X1,i-j),np.power(X2,j))]
    return out
#function for cost with regularisation    
def costReg(theta,X,y,lamb):
    XbyT = X*theta
    return ((-y.T*np.log(sigmoid(XbyT))-(1-y).T*np.log(1-sigmoid(XbyT)))+theta[1:,:].T*theta[1:,:]*lamb/2)/y.shape[0]
#function for calculating gradient this will be used in gdbt
def gradientReg(theta,X,y,lamb):
    sigXbyT = sigmoid(X*theta)
    grad = np.matrix(np.zeros(theta.shape[0])).T
    grad[0] = X[:,0].T*(sigXbyT-y)
    grad[1:] = X[:,1:].T*(sigmoid(X*theta)-y)+lamb*theta[1:]
    return grad/y.shape[0]
    
def gdbt(theta,X,y,lamb,alpha,beta,iter,tol):
   
    # Input:
        # theta: Initial value
        # X: Training data (input)
        # y: Training data (output)
        # alpha: Parameter for line search, denoting the cost function will be descreased by 100xalpha percent
        # beta: Parameter for line search, denoting the "step length" t will be multiplied by beta
        # iter: Maximum number of iterations
        # tol: The procedure will break if the square of the 2-norm of the gradient is less than the threshold tol
    for i in range(iter):
        grad = gradientReg(theta,X,y,lamb)
        delta = -grad
        if grad.T*grad < tol:
            print 'Terminated due to stopping condition with iteration number',i
            return theta
        J = costReg(theta,X,y,lamb)
        alphaGradDelta = alpha*grad.T*delta
        # begin line search
        t = 1
        while costReg(theta+t*delta,X,y,lamb) > J+t*alphaGradDelta:
            t = beta*t
        # end line search

        # update value of theta after minimisation
        
        theta = theta+t*delta
    return theta
#function for plotting decision boundry based on polynomial
def plotDecisionBoundary(X,degree):
    x1Array = np.linspace(np.min(X[:,1]),np.max(X[:,1]),num=50)
    x2Array = np.linspace(np.min(X[:,2]),np.max(X[:,2]),num=50)
    z = np.zeros((len(x1Array),len(x2Array)))
    X1,X2 = np.meshgrid(x1Array,x2Array)
    for i in range(len(x1Array)):
        for j in range(len(x2Array)):
            z[i,j] = mapFeature(np.matrix(x1Array[i]),np.matrix(x2Array[j]),degree)*theta
    
    plt.contour(x1Array,x2Array,z,1)
    from mpl_toolkits.mplot3d import Axes3D

    from matplotlib import cm

    fig1 = plt.figure()

    ax1 = fig1.add_subplot(1,1,1,projection='3d')

    surf = ax1.plot_surface(X1, X2, z)

    ax1.set_xlabel("Test 1")

    ax1.set_ylabel("Test 2")

    ax1.set_zlabel("h_theta")

    cset = ax1.contour(X1, X2, z, levels=[5], zdir='z', offset=5, cmap=cm.coolwarm)



#standard predict function as used in logistic regression,remeber h suses sigmoid function    
def predict(theta,X):
    return np.matrix([1 if sigmoid(x*theta)>=0.5 else 0 for x in X]).T
    
# read date and process for logistic regression
f = open('ex2data2.txt')
X = []
y = []
for line in f:
    data = line.split(',')
    X.append([float(data[0]),float(data[1])])
    y.append(float(data[2]))
#ploting newly read data
plotData(X,y)
plt.show()
plotData(X,y)
X = np.matrix(X)
y = np.matrix(y).T
#degree of polynomial
degree = 6

# feature mapping
X = mapFeature(X[:,0],X[:,1],degree)
#initial theata initialisation
initial_theta = np.matrix(np.zeros(X.shape[1])).T
#cost and gradient for initial values of theta
cost = costReg(initial_theta,X,y,1)
gradient = gradientReg(initial_theta,X,y,1)

print 'Cost at initial theta (zeros):',cost


#regularisation constant
lamb = 10
#leaning rate
alpha = 0.01
beta = 0.8
#performing logistic regression 
theta = gdbt(initial_theta,X,y,lamb,alpha,beta,1000,1e-8)
#ploting contor
plotDecisionBoundary(X,degree)
#p is a matrix with predictions 
p = predict(theta,X)

print 'Train accuracy:',(len(y)-np.sum(np.abs(y-p)))/len(y)*100,'%'


plt.show()


