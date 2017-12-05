import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

####################
# Fonctions utiles #
####################

def drawInputsInSquare(n):
    """
    Generate n points in the square [0,1] X [0,1]
    as a matrix of size (n,2)
    """
    return np.random.random((n,2))

def addConstantInput(X):
    """
    This function returns a copy of X with an extra column full of ones
    """
    n,m = X.shape
    X2 = np.zeros((n,m+1))
    X2[:,:-1] = X
    X2[:,-1] = np.ones(n)
    return np.asmatrix(X2)
    
def drawOutput(X, theta, sigmae):
    """
    Generate outputs for inputs in matrix X
    using model provided by theta = [theta0, theta1, theta2]
    and a normal white noise of standard deviation sigmae
    """
    n = X.shape[0]
    theta = np.asmatrix(theta).reshape((3,1))
    Y = X * theta + np.random.normal(0,sigmae, size = (n,1))
    return Y

def drawModel(ax, theta, c):
    """
    Draw the plane of predictions for parameters theta
    """
    theta = np.asarray(theta)
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = theta[0]*X+theta[1]*Y+theta[2]
    return ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color = c)

def quadraticRisk(Y1,Y2):
    """
    Compute the quadratic risk between two outputvectors of same dimension
    """
    assert(Y1.shape == Y2.shape)
    n = Y1.shape[0]
    D = Y1 - Y2
    return 1./n * (D.T * D)[0,0]

def compareModels(realTheta, estTheta, X, realY):
    """
    Draw the estimated and real models (i.e. plane)
    - realTheta and estTheta are 3-arrays of type [theta0, theta1, theta2]
    - If inputs X and outputs realY are provided, real points and estimated points (for the same inputs) are also drawn. Empirical risk is returned.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    drawModel(ax, realTheta, 'blue')
    drawModel(ax, estTheta, 'red')
    risk = None
    if(not(X is None) and not(realY is None)):
       ax.scatter(np.asarray(X[:,0]), np.asarray(X[:,1]), np.asarray(realY), c = 'blue', marker = 'o')
       estY = X * np.asmatrix(np.reshape(estTheta, (3,1)))
       risk = quadraticRisk(realY, estY)
       plt.suptitle("Quadratic Risk J = {:.4f}".format(risk))
       ax.scatter(np.asarray(X[:,0]), np.asarray(X[:,1]), np.asarray(estY), c = 'red', marker = '+')

       X = np.asarray(X)
       n = X.shape[0]
       realY = np.asarray(realY).reshape(n)
       estY = np.asarray(estY).reshape(n) 
       for (x,yr,yp) in zip(X,realY,estY):
          ax.plot([x[0], x[0]], [x[1], x[1]], [yr, yp], label = 'red')
       print("Quadratic Risk J = {:.4f}".format(risk))
       
    blue_patch = mpatches.Patch(color='blue', label='Real')
    red_patch = mpatches.Patch(color='red', label='Estimation')
    plt.legend(handles=[red_patch, blue_patch])
    plt.draw()
    return risk

def testRegressor(realTheta, n, sigmae, regressor, inputGenerator):
    '''
    Estimate a regressor on n samples (provided by function regressor)
    whose input are drawn from generator function inputGenerator
    and whose output is drawn from model (realTheta, sigmae)
    Then graphically shows the result of estimation.
    '''
    Xtrain = inputGenerator(n)
    Xtrain = addConstantInput(Xtrain)
    Ytrain = drawOutput(Xtrain, realTheta, sigmae)
    theta = regressor(Xtrain,Ytrain)
    risk = compareModels(realTheta, theta, Xtrain, Ytrain)
   
def drawTrainAndTestEmpiricalRisk(nSamples, sigmas, Jtrain, Jtest):
    """
    Draws colormaps of square roots of two empirical risks, one computed on a train dataset, the other on a test dataset
    Risks must be computed for each couple of (number of samples, value of error standard deviation
    Vector nSamples contains the set of number of samples
    Vector sigmas contains the set of values of error standard deviation
    Sizes of matrices Jtrain and Jtest must be compatible with vectors nSamples and sigmas
      - Number of rows of Jtrain and Jtest must be equal to the length of sigmas
      - Number of columns of Jtrain and Jtest must be equal to the length of nSamples
    """
    XX,YY = np.meshgrid(nSamples,sigmas)
    fig, axarr = plt.subplots(2, sharex=True)
    maxJ = np.sqrt(np.mean(Jtrain)) * 10.
    plt.subplot(2, 1, 1)
    plt.pcolor(XX,YY,np.sqrt(Jtrain),vmin=0, vmax=maxJ)
    plt.colorbar()    
    plt.title('Square root of empirical risk of train data set')
    plt.subplot(2, 1, 2)
    plt.pcolor(XX,YY,np.sqrt(Jtest),vmin=0, vmax=maxJ)
    plt.colorbar()    
    plt.title('Square root of empirical risk of test data set')
    plt.draw()
    
def drawInputsAlmostAligned(n, alignmentFactor = 0.1):
    """
    Draws n points almost aligned on the first diagonal
    alignmentFactor is an alignment factor. alignmentFactor = 0 gives perfectly aligned points.
    """
    alpha = np.pi/4
    ca = np.cos(alpha); sa = np.sin(alpha)
    A = (np.random.random((n,2)) - 0.5)
    B = A * np.matrix([[1., 0.], [0., alignmentFactor]])
    X = B * np.matrix([[ca, sa], [-sa, ca]]) + np.array([0.5,0.5])
    return X

def plotFilterState(theta, sigmas):
    '''
    Plot the three estimated theta coefficients as function of time.
    - theta must be a matrix of size (3,n) where i-th column contains estimated parameters at i-th iteration.
    - sigmas must be a matrix of size (3,n) where i-th column contains estimated variances of each parameter at i-th iteration.
    Covariances contained in sigmas are used to draw confidence intervals of 95% around expected parameter values mu, 
    equal to [ mu - 2*sigma, mu + 2*sigma].
    '''
    fig = plt.figure()
    n = theta.shape[1]
    for i in range(3):
      plt.subplot(3,1,i+1)
      plt.plot(theta[i,:],'r')
      intervalTop = theta[i,:] + 2 * sigmas[i,:]
      intervalBottom = theta[i,:] - 2 * sigmas[i,:]
      plt.fill_between(range(n), intervalBottom, intervalTop, facecolor = 'lightgray')
      
      plt.plot(intervalTop,'--')
      plt.plot(intervalBottom,'--')
      plt.xlabel('$\\theta_{}$'.format(i))
      ymean = np.mean(theta[i,:])
      sigmaMean = np.median(sigmas[i,:])
      axes = plt.gca()
      axes.set_ylim([ymean - 4 * sigmaMean, ymean + 4 * sigmaMean])

def drawOutputWithDrift(X, theta0, sigmae, sigmac):
    """
    Generate outputs for inputs in matrix X with drift in the model coefficients
    model coefficients are initialized with theta0 and then
    drift with a normal white noise of standard deviation sigmac
    output has an additive normal white noise of standard deviation sigmae

    Returns (Y,theta)
    Y is a n-column vector containing the outputs (n is the number of lines of X)
    theta is a nx3 matrix where line i gives the parameters used to generate Y[i]
    """
    n = X.shape[0]
    X = np.asarray(X)
    theta = np.zeros((n,3))
    for i in range(0,n):
        theta[i,:] = theta0
        if(sigmac > 0.):
            theta0 += np.random.normal(0, sigmac, size = 3)
    Y = np.sum(X * theta,1) + np.random.normal(0,sigmae, size = (n,))
    Y = np.reshape(Y,(n,1))
    return (Y, theta)

def evaluateRegressor(regressor,inputGenerator,realTheta,sigmas,nSamples):
    Jtrain = np.zeros((len(sigmas),len(nSamples)))
    Jtest = np.zeros((len(sigmas),len(nSamples)))
    K = 20
    nTest = 1000

    for k in range(K):
      print("Pass {}/{}      \r".format(k+1,K), end='')
      Xtest = drawInputsInSquare(nTest)
      Xtest = addConstantInput(Xtest)
      for i,sigma in enumerate(sigmas):
        Ytest = drawOutput(Xtest, realTheta, sigma)
        for j,n in enumerate(nSamples):
           Xtrain = inputGenerator(n)
           Xtrain = addConstantInput(Xtrain)
           Ytrain = drawOutput(Xtrain, realTheta, sigma)
           theta = regressor(Xtrain,Ytrain)
           Jtrain[i,j] += quadraticRisk(Ytrain, Xtrain * theta)
           Jtest[i,j] += quadraticRisk(Ytest, Xtest * theta)
    Jtrain /= K
    Jtest /= K   
    drawTrainAndTestEmpiricalRisk(nSamples, sigmas, Jtrain, Jtest)
      
####################
# Code à compléter #
####################

# Question 1.2

def computeOLS(X,Y):
    '''
    Return the standard OLS estimator computed from X and Y
    '''
    thetaOLS = (X.T*X).I*X.T*Y
    return thetaOLS

def testSimpleOLS(n, sigmae):
    print('Question 1.2: Simple OLS')
    realTheta = np.array([1,2,3])
    testRegressor(realTheta,n,sigmae,computeOLS, drawInputsInSquare)
# Question 1.3

def evaluateSimpleOLS():
    print('Question 1.3: Simple OLS')
    realTheta = np.matrix([1,2,3]).T
    sigmas = np.arange(0.5,30,0.5)
    nSamples = np.arange(3,20,1)
    evaluateRegressor(computeOLS, drawInputsInSquare, realTheta, sigmas, nSamples)
    
# Question 1.5           

def testSimpleOLSWithAlignedPoints(n, sigmae, alignmentFactor):
    print('Question 1.5: Simple OLS with almost aligned input points')
    def drawInput(X): return drawInputsAlmostAligned(X,alignmentFactor)
    realTheta = np.matrix([1,2,3]).T
    sigmas = np.arange(0.5, 30, 0.5)
    nSamples = np.arange(3, 20, 1)    
    testRegressor(realTheta, n, sigmae, computeOLS, drawInput)
    evaluateRegressor(computeOLS, drawInput, realTheta, sigmas, nSamples)
    

# Question 2.11
def computeRidge(lambdaFactor,X,Y):
    n=X.shape[1]
    theta = (X.T*X+lambdaFactor*np.eye(n)).I*X.T*Y
    return theta

def testRidgeRegression(n, sigmae, alignmentFactor, lambdaFactor):
    print('Ridge regression with almost aligned input points')
    def drawInput(X): return drawInputsAlmostAligned(X,alignmentFactor)
    def ridgeRegression(X,Y): return computeRidge(lambdaFactor,X,Y)

    realTheta = np.array([1,2,3])
    sigmas = np.arange(0.5, 30, 0.5)
    nSamples = np.arange(3, 20, 1)    
    testRegressor(realTheta, n, sigmae, computeOLS, drawInput)
    testRegressor(realTheta, n, sigmae, ridgeRegression, drawInput)
    evaluateRegressor(computeOLS, drawInput, realTheta, sigmas, nSamples)
    evaluateRegressor(ridgeRegression, drawInput, realTheta, sigmas, nSamples)


# Question 3.14

class Filter:
    def __init__(self, sigma0):
       '''
       Constructor of a Kalman fitler for recursive least square
       - self is a reference to the current object ("self" in Python is like "this" in Java)
       - Each state component is supposed to be known with an uncertainty given by the same initial standard deviation sigma0.
       '''
       self.state = 0
       self.P = 0
       
    def update(self, y, x, sigmaY):
       '''
       Function called to update the filter state when a new observation (x,y) is received.
       - self is a reference to the current object ("self" in Python is like "this" in Java)
       - sigmaY is the standard deviation of the observation noise.
       '''
       self.state = 0
       self.P = 0

    def integrate(self, Q):
       '''
       Function called to integrate the filter state every second.
       - self is a reference to the current object ("self" in Python is like "this" in Java)
       - Q is a square matrix containing the state noise integrated over one second
       '''
       self.state = 0
       self.P = 0
    
def testRecursiveLeastSquare(n, sigmae):
    print('Recursive Least Square')
    
    realTheta = np.matrix([1. , 2., 3.]).T
    Xtrain = drawInputsInSquare(n)
    Xtrain = addConstantInput(Xtrain)
    Ytrain = drawOutput(Xtrain, realTheta, sigmae)
    
    thetaEst = np.zeros((3,n))
    sigmaEst = np.zeros((3,n))
    filter = Filter(1.E3)
    
    for t in range(n):
       filter.update(Ytrain[t], Xtrain[t,:], sigmae)
       thetaEst[:,t] = np.reshape(filter.state,3)
       sigmaEst[:,t] = np.sqrt(np.diag(filter.P))

    plotFilterState(thetaEst, sigmaEst)
    
    thetaOLS = computeOLS(Xtrain,Ytrain)
    print("Final state : {}".format(filter.state.T))
    print("OLS estimate: {}".format(thetaOLS.T))

# Question 3.15

def testKalman(n, sigmae, sigmac):
    print('Kalman filter with integration')
    
    theta0 = np.array([1., 2., 3.])
    Xtrain = drawInputsInSquare(n)
    Xtrain = addConstantInput(Xtrain)
    Ytrain, theta = drawOutputWithDrift(Xtrain, theta0, sigmae, sigmac)
    thetaEst = np.zeros((3,n))
    sigmaEst = np.zeros((3,n))
    filter = Filter(1.E3)
    Q = np.eye(3) * (sigmac ** 2)

    for t in range(n):
       filter.integrate(Q)
       filter.update(Ytrain[t], Xtrain[t,:], sigmae)
       thetaEst[:,t] = np.reshape(filter.state,3)
       sigmaEst[:,t] = np.sqrt(np.diag(filter.P))

    plotFilterState(thetaEst, sigmaEst)
    
    thetaOLS = computeOLS(Xtrain,Ytrain)
    print("Final real state     : {}".format(theta[-1,:]))
    print("Final estimated state: {}".format(filter.state.T))
    print("OLS estimate         : {}".format(thetaOLS.T))
  

########
# Main #
########

def main():
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default = 1,
	            help="procedure to run")
    parser.add_argument("--n", type=int, default = 100,
	            help="number of generated samples")
    parser.add_argument("--sigma", type=float, default = 1.,
	            help="standard deviation of error")
    parser.add_argument("--alignment", type=float, default = 0.1,
	            help="alignment factor")
    parser.add_argument("--lambda", type=float, default = 1.,
	            help="lambda factor")
    parser.add_argument("--drift", type=float, default = 1.,
	            help="model coefficient drift (standard deviation)")
    args = parser.parse_args()
    
    if(args.test == 1):
        testSimpleOLS(args.n, args.sigma)
    elif(args.test == 2):
        evaluateSimpleOLS()
    elif(args.test == 3):
        testSimpleOLSWithAlignedPoints(args.n, args.sigma, args.alignment)
    elif(args.test == 4):
        testRidgeRegression(args.n, args.sigma, args.alignment, getattr(args, 'lambda'))
    elif(args.test == 5):
        testRecursiveLeastSquare(args.n, args.sigma)
    elif(args.test == 6):
        testKalman(args.n, args.sigma, args.drift)
    plt.show()
if __name__ == "__main__":
    main()
