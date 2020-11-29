# Plot 2D-feature data to be classified
```
import matplotlib.pyplot as plt
%matplotlib inline

pos = y==1
neg = y==0

plt.plot(X[pos, 0], X[pos, 1], 'k+')
plt.plot(X[neg, 0], X[neg, 1], 'wo', mec='k')

plt.xlabel("x_1")
plt.ylabel("x_2")
plt.legend(['Positive Class', 'Negative Class'])
```  
<br>


# Advanced Optimization for Logistic Reg
![](http://mathurl.com/render.cgi?h_%5Ctheta%28x%29%3Dg%28%5Ctheta%5ETx%29%5Cquad%5Ctext%7Bwhere%7D%5Cquad%20g%28z%29%20%3D%20%5Cfrac%7B1%7D%7B1+e%5E%7B-z%7D%7D%5Cnocache)   
![](http://chart.apis.google.com/chart?cht=tx&chl=J(\theta)=\frac{1}{m}\sum_{i=1}^{m}\left[-y^{(i)}\log\left(h_\theta\left(x^{(i)}\right)\right)-\left(1-y^{(i)}\right)\log\left(1-h_\theta\left(x^{(i)}\right)\right)\right])   
![](http://mathurl.com/render.cgi?%5Cfrac%7B%5CpartialJ%28%5Ctheta%29%7D%7B%5Cpartial%5Ctheta_j%7D%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%5Cleft%28h_%5Ctheta%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29-y%5E%7B%28i%29%7D%5Cright%29x_j%5E%7B%28i%29%7D%5Cqquad%5Ctext%7Bfor%7D%20j%20%3D%200%2C%201%2C%20%5Ccdots%20%2C%20n%20%5Cnocache)   

## a) Prep
```
# Define sigmoid function
def sigmoid(z):
    """
    Compute sigmoid function g(z) given input z.

    Parameters
    ----------
    z : array_like
        The input to the sigmoid function. This can be a 1-D vector 
        or a 2-D matrix.

    Returns
    ----------
    g : array_like
        The computed sigmoid function. g has the same shape as z, since
        the sigmoid is computed element-wise on z.
    """
    z = np.array(z)
    g = np.zeros(z.shape)

    import math
    g += 1/(1 + math.e**(-1 * z))

    return g
```   
```
# Write a fucntion to return cost and gradient (the two inputs of particular advanced optiminzation method)
def costFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression. 
    
    Parameters
    ----------
    theta : array_like
        Parameters for logistic regression. Vector
        of shape (n+1, ).
    
    X : array_like
        Input dataset of shape (m x n+1). We assume the 
        intercept has already been added to the input.
    
    y : array_like
        Labels for the input. Vector of shape (m, ).
    
    Returns
    -------
    J : float
        The computed value for the cost function. 
    
    grad : array_like
        A vector of shape (n+1, ) which is the gradient of the cost
        function with respect to theta.
    """
    J = 0
    grad = np.zeros(theta.shape)
    
    m = y.size

    # Compute cost
    h = sigmoid(np.dot(X, theta))
    Cost = np.multiply((-1 * y), np.log(h)) - np.multiply((1-y), np.log(1-h))
    J += (1/m) * Cost.sum()

    # Compute gradient
    grad += (1/m) * np.dot((h - y).transpose(), X)

    return J, grad
```   
## b) Optimization
```
# Use scipy.optimize module to perform optimization
from scipy import optimize

options = {'maxiter' : n}           # <- set max # iterations (int) performed by optimization

# Assign OptimizationResult object
res = optimize.minimize(costFunction,       # <- cost function that computes the logistic regression cost and gradient for training set
                        initial_theta,      # <- initial values of theta; predefined array (often of zeros)
                        (X, y),             # <- pre-defined arguments for cost function; other necessary args not including theta
                        jac=True,           # <- indication that cost function also returns the Jacobian (gradient)
                        method='TNC',       # <- using truncated Newton algorithm for optimization
                        options=options)    # <- additional options

cost = res.fun    # <- cost at optimized theta
theta = res.x     # <- optimized theta
```
<br>


# Advanced Optimization for Regularized Logistic Regression
![](http://mathurl.com/render.cgi?h_%5Ctheta%28x%29%3Dg%28%5Ctheta%5ETx%29%5Cquad%5Ctext%7Bwhere%7D%5Cquad%20g%28z%29%20%3D%20%5Cfrac%7B1%7D%7B1+e%5E%7B-z%7D%7D%5Cnocache)   
![](http://mathurl.com/render.cgi?%24%24%20J%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5Em%20%5Cleft%5B%20-y%5E%7B%28i%29%7D%5Clog%20%5Cleft%28%20h_%5Ctheta%20%5Cleft%28x%5E%7B%28i%29%7D%20%5Cright%29%20%5Cright%29%20-%20%5Cleft%28%201%20-%20y%5E%7B%28i%29%7D%20%5Cright%29%20%5Clog%20%5Cleft%28%201%20-%20h_%5Ctheta%20%5Cleft%28%20x%5E%7B%28i%29%7D%20%5Cright%29%20%5Cright%29%20%5Cright%5D%20+%20%5Cfrac%7B%5Clambda%7D%7B2m%7D%20%5Csum_%7Bj%3D1%7D%5En%20%5Ctheta_j%5E2%20%24%24%0A%0A%24%24%20%5Cfrac%7B%5Cpartial%20J%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta_0%7D%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5Em%20%5Cleft%28%20h_%5Ctheta%20%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29%20-%20y%5E%7B%28i%29%7D%20%5Cright%29%20x_j%5E%7B%28i%29%7D%20%5Cqquad%20%5Ctext%7Bfor%20%7D%20j%20%3D0%20%24%24%0A%0A%24%24%20%5Cfrac%7B%5Cpartial%20J%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta_j%7D%20%3D%20%5Cleft%28%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5Em%20%5Cleft%28%20h_%5Ctheta%20%5Cleft%28x%5E%7B%28i%29%7D%5Cright%29%20-%20y%5E%7B%28i%29%7D%20%5Cright%29%20x_j%5E%7B%28i%29%7D%20%5Cright%29%20+%20%5Cfrac%7B%5Clambda%7D%7Bm%7D%5Ctheta_j%20%5Cqquad%20%5Ctext%7Bfor%20%7D%20j%20%5Cge%201%20%24%24%5Cnocache)  
## a) Prep
```
# Write a function to return cost and gradient (the two inputs of particular advanced optiminzation method)
def costFunctionReg(theta, X, y, lambda_):
    """
    Compute cost and gradient for regularized logistic regression. 
    
    Parameters
    ----------
    theta : array_like
        Parameters for logistic regression. Vector
        of shape (n+1, ).
    
    X : array_like
        Input dataset of shape (m x n+1). We assume the 
        intercept has already been added to the input.
    
    y : array_like
        Labels for the input. Vector of shape (m, ).

    lambda_ : float
        The regularization parameter. 
    
    Returns
    -------
    J : float
        The computed value for the cost function. 
    
    grad : array_like
        A vector of shape (n+1, ) which is the gradient of the cost
        function with respect to theta.
    """
    m = y.size

    J = 0
    grad = np.zeros(theta.shape)

    # Compute cost
    h = sigmoid(np.dot(X, theta))
    Cost = np.multiply((-1 * y), np.log(h)) - np.multiply((1-y), np.log(1-h))

    Cost_regterm = (lambda_/(2*m)) * np.dot(theta[1:], theta[1:])   # <- regularization term doesn't penalize theta_0
    J += (1/m) * Cost.sum() + Cost_regterm

    # Compute gradient/jacobian
    diagonal = np.eye(theta.size, theta.size)     # <- (n + 1, n + 1) special matrix
    diagonal[0,0] = 0

    grad_regterm = (lambda_/m) * np.dot(diagonal, theta)   # np.dot(...) replaces theta_0 in theta vector with 0 so gradient of theta_0 isn't penalized by regularization term

    g += (1/m) * np.dot((h - y).transpose(), X) + grad_regterm

    return J, grad
```
>rather than using special matrix can also just copy theta column vector and set 0,0 index to 0

 ## b) Optimization
 ```
 from scipy import optimize

options = {'maxiter' : n}   # <- set max # iterations (int) performed by optimization

# Assign OptimizationResult object
res = optimize.minimize(costFunctionReg,   # <- cost function that computes regularized logistic regression cost and gradient for training set
                        initial_theta,     # <- initial values of theta; predefined array (often of zeros)
                        (X, y, lambda_),   # <- pre-defined arguments for cost function; other necessary args not including theta
                        jac=True,          # <- indication that cost function also returns the Jacobian (gradient)
                        method='TNC',      # <- using truncated Newton algorithm for optimization
                        options=options)   # <- additional options

cost = res.fun   # <- access cost at optimized theta
theta = res.x    # <- access array of optimized theta values
```  
<br>


# Logistic Reg One-class classification Predictions
```
z = np.dot(X, theta)     # <- X is array_like w/ shape (m, n+1)
yhat = sigmoid(z)

m = X.shape[0]
p = np.zeros(m)
p += np.round(yhat)       # => output of 0 and 1 (what we expect y to equal)
```
 
