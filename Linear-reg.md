# Convert df to array
```
    import pandas as pd
    data_array = pd.DataFrame.to_numpy(data)    # <- where data is pandas DataFrame
```
After assigning X and y ensure that both are of type 'float'
```
    X = X.astype('float')
    y = y.astype('float)
```
<br>

# Mean Squared Error Cost Function
![](http://chart.apis.google.com/chart?cht=tx&chl=J(\theta)=\frac{1}{2m}\sum_{i=1}^m\left(h_{\theta}(x^{(i)})-y^{(i)}\right)^2)
```
def computeCost(X, y, theta):
    """
    Vectorized code to compute cost for linear regression
    Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    Returns
    -------
    J : float
        The value of the cost function. 
    """
    m = y.shape[0]      # number of training examples
    
    J = 0

    h = np.dot(X, theta)
    error = h - y

    J += (1/(2*m)) * ((error**2).sum())
    
    return J
```
<br>

# Gradient Descent
_repeat to convergence {_  
![](http://chart.apis.google.com/chart?cht=tx&chl=\theta_j:=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^m\left(h_\theta(x^{(i)})-y^{(i)}\right)x_j^{(i)}\qquad\qquad) simultaneously update ![](http://chart.apis.google.com/chart?cht=tx&chl=\theta_j) for all j  
_}_  
```
def gradientDescent(X, y, theta):
    """
    Vectorized code to perform gradient descent to best theta values.
    Updates theta by taking num_iters gradient steps with learning rate alpha.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    alpha : float
        The learning rate for gradient descent. 
    
    num_iters : int
        The number of iterations to run gradient descent. 
    
    Returns
    -------
    best_theta : array_like
        The learned, "best" linear regression parameters. A vector of shape (n+1, ).
    
    J_history : list
        A python list for the values of the cost function after each iteration. Uses computeCost() to calculate J.
    """

    m = y.shape[0]     # number of training examples
    
    # make a copy of theta, so as to not directly alter initialized values
    theta = theta.copy()
    
    J_history = []
    
    for i in range(num_iters):
        h = np.dot(X, theta)
        error = h - y
        delta = (1/m) * np.dot(error.transpose(), X)
        change = alpha * delta
        theta -= change.transpose()
        
        # save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))
    
    return theta, J_history
```
* ## selecting alpha   
```
alpha = [.001, .003, .01, .03, .1, .3]

n = X.shape[1]    # number features 
num_iters = 50

for num in alpha:
    # conduct gradient descent using gradientDescent()
    best_theta, J_history = gradientDescent(X=X, y=y, theta=np.zeros((n)), alpha=num, num_iters=num_iters)    # <- using feat. normalized X

    # DEBUGGING; plots curve of cost vs. number iterations using J_history values
    plt.plot(range(num_iters), J_history, '-', label=num)
    plt.xlabel("# iterations")
    plt.ylabel("Cost (J)")
    plt.legend()
```

* ## feature normalization example (mean normalization)  
**After** performing feature normalization, reassign x_0 values (vector of 1's) and normalized values to X:
```
    m = X.shape[0]
    X = np.concatenate(np.ones(m), X_norm, axis=1)
```
```
def featureNormalize(X):
    """
    Vectorized code performs mean normalization, (x-mu)/s, on the features in X. Returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation is 1.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).
    
    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).
    
    mu : array_like
        A vector of shape (n, ) for the mean values of each feature.

    sigma : array_like
        A vector of of shape (n, ) for standard deviation values of each feature.
    """

    X_norm = X.copy()    # make a copy of X to avoid directly altering original data

    mu = np.mean(X_norm, axis=0)     # average the rows of each column
    sigma = np.std(X_norm, axis=0)  

    m = X_norm.shape[0]

    full_mu = np.outer(mu, np.ones((1, m))).transpose()        
    full_sigma = np.outer(sigma, np.ones((1, m))).transpose()

    X_norm = np.divide((X_norm - full_mu), full_sigma)
    
    return X_norm, mu, sigma
```
<br>

# Normal Equation 
_Analytically solve for best theta values for linear reg_  
![](http://chart.apis.google.com/chart?cht=tx&chl=\theta=\left(X^TX\right)^{-1}X^Ty)  
```
def normalEqn(X, y):
    """
    Vectorized code computes the closed-form solution to linear regression using the normal equations.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        The value at each data point. A vector of shape (m, ).
    
    Returns
    -------
    theta : array_like
        Estimated linear regression parameters. A vector of shape (n+1, ).
    
    """
    theta = np.zeros(X.shape[1])
    
    theta += np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), np.dot(X.transpose(), y))
    
    return theta
```
<br>

# Plotting h(x) regression line
```
# plot data points
plt.plot(x, y)

# plot regression line
plt.plot(X[:, 1], np.dot(X, theta), '-')     # np.dot(X, theta) => h(x); input data is X[:, 1]; don't include feature0 here
plt.legend(['Training data', 'Linear regression'])  

pyplot.show()
```
<br>

# Make predictions using h(x)
Recall ![](http://chart.apis.google.com/chart?cht=tx&chl=h_\theta(x)=\theta_0x_0+\theta_1x_1+...+\theta_nx_n) 
 
* ## without normalization 
```
Input = [1, x_1, x_2, ...]      # <- where 1, x_1, ... are input values for feature0, feature1, ... 

y_hat = np.dot(Input, theta)    # <- where theta is vector [theta_0, theta_1, theta_2, ...]
```  
* ## with normalization example (mean normalization)
```
prenormInput = [x_1, x_2, ...]
postnormInput = np.divide((prenormInput - mu), sigma)   # <- where mu and sigma are vectors returned by featureNormalize()

Input = np.append([1], postnormInput)                   # <- append normalized inputs to 1

y_hat =  np.dot(Input, theta)
``` 
