# Regularized Logistic Reg Multi-class Classification Predictions

## a) Optimize parameters
```
def oneVsAll(X, y, num_labels, lambda_):
    """
    Trains logistic regression classifier for each class 
    and returns parameters for each classifier in matrix all_theta
    where i-th row corresponds to parameters of classifier for class i

    Parameters
    -----------
    X : array_like
        Input dataset, matrix of shape (m, n).
        Notice the X we pass does not have x_0 column of ones

    y : array_like
        Labels of input dataset, column vector of shape (m, )

    num_labels : int
        Number of possible labels (aka number of classes, K)

    lambda_ : float
        Logistic Regression regularization parameter, passed to 
        CostFunction() used by scipy's optimize.minimize()

    Returns
    -----------
    all_theta : array_like
        Parameters for each classifier, matrix of shape (K, n+1) 
        where K is number of classes
    """
    m, n = X.shape
    X = np.concatenate((np.ones((m,1)), X), axis=1)   # add x_0 column to input dataset => X of shape (m, n+1)

    all_theta = np.zeros((num_labels, n+1))


    # use scipy's optimize.minimize() to find parameters for each classifier
    options = {'maxiter' : n}     # n = maximum number of iterations for optimize.minimize()
    initial_theta = np.zeros(n+1)

    for i in range(num_labels):
        res = optimize.minimize(CostFunction,         # <- predefined Regularized Linear Regression Cost function with returns J and gradient
                                initial_theta,
                                (X, y==i, lambda_),   # <- args other than initial_theta needed by CostFunction; one-vs-all so y should be matrix of 0's and 1's (where examples of every other class besides class of interest are labelled 0)
                                jac=True,             # CostFucntion() returns jacobian (gradient)
                                method = 'CG',        # NOTE: using CG here not TNC
                                options=options)       
        theta = res.x
        all_theta[i, :] += theta.transpose().tolist()    # <- store classifier's optimized parameters in corresponding row of all_theta
        # NOTE: res.x returns array; all_theta is an array (list of lists) therefore must overwrite row using a list

    return all_theta
```
## b) Make predictions
```
def predictOneVsAll(X, all_theta):
    """
    Return a vector of predictions for each example in the matrix X.
    For multiclass classification using regularized logistic regression.

    Parameters
    -----------
    X : array_like
        Input datapoints for prediction, of shape (m, n)

    all_theta : array_like
        Parameters for each regularized logistic regression 
        classifier, matrix of shape (K, n+1) where K is 
        number of classes

    Returns
    -----------
    p : array_like
        Predictions for each data point (m) in X, of shape (m, ) 
    """
    m, n = X.shape
    p = np.zeros(m)

    Input = np.concatenate((np.ones((m,1)), X), axis=1)   # <- add x_0 feature of 1's

    # put each test example through each classifier, returns matrix of shape (m, K)
    h = sigmoid(np.dot(Input, all_theta.transpose()))   # <- NOTE: sigmoid is pre-defined function
    
    # return index (aka class) of maximum h 
    p += np.argmax(h, axis=1)     # <- use axis=1 because comparing numbers which are side-by-side/in adjacent columns


    return p
```
<br>


# Neural Network Forward Propogation 
## One-vs-all Multiclass Classification Prediction
![](http://mathurl.com/render.cgi?%5Ctext%7Bfor%20each%20node%20k%7D%20%5Cquad%20z_k%5E%7B%28j%29%7D%20%3D%20%5CTheta_%7B0%2Ck%7D%5E%7B%28j-1%29%7D%20%5Ccdot%20a_0%5E%7B%28j-1%29%7D%20+%20%20%5CTheta_%7B1%2Ck%7D%5E%7B%28j-1%29%7D%20%5Ccdot%20a_1%5E%7B%28j-1%29%7D%20+%20...%20+%20%5CTheta_%7Bn%2Ck%7D%5E%7B%28j-1%29%7D%20%5Ccdot%20a_n%5E%7B%28j-1%29%7D%5Cnocache)  
![](http://mathurl.com/render.cgi?a%5E%7B%28j-1%29%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20x_0%20%5C%5C%20%5Cvdots%20%5C%5C%20x_n%20%5Cend%7Bbmatrix%7D%5Cnocache)  
![](http://mathurl.com/render.cgi?%24z%5E%7B%28j%29%7D%20%3D%20%5CTheta%5E%7B%28j-1%29%7D%20%5Ccdot%20a%5E%7B%28j-1%29%7D%20%5Cqquad%20a%5E%7B%28j%29%7D%20%3D%20g%28z%5E%7B%28j%29%7D%29%24%20%0A%0A%24z%5E%7B%28j+1%29%7D%20%3D%20%5CTheta%5E%7B%28j%29%7D%20%5Ccdot%20a%5E%7B%28j%29%7D%20%5Cqquad%20h_%7B%5CTheta%7D%28x%29%20%3D%20a%5E%7B%28j+1%29%7D%20%3D%20g%28z%5E%7B%28j+1%29%7D%29%24%5Cnocache)  
```
def predict(X, Theta1, Theta2):
    """
    Predicts the label of an input given a trained neural network. 

    Parameters
    ----------
    X : array_like
        Dataset of image inputs having shape (m, image dimensions).
        Ex: if image is a x b pixels, dimensions/# input fetaures = a*b

    Theta1 : array_like
        Weights going from the first layer (j-1) into the second layer (j) of
        the neural network. Has shape (s_j, s_(j-1) + 1) where s_x is the
        number of nodes in layer x of a neural network.

    Theta2 : array_like
        Weights going from the second layer (j) into the third layer (j+1) of
        the neural network. Has shape (s_(j+1), s_j + 1) where s_x is the
        number of nodes in layer x of a neural network.


    Returns
    ----------
    p : array_like
        Vector containing the predicted label for each example.
        Has shape (m, ).

    """
    m, n = X.shape
    p = np.zeros(m)
    
    X = np.concatenate((np.ones((m, 1)), X), axis=1)  # <- add bias unit to input layer
    a1 = X.transpose()   

    z2 = np.dot(Theta1, a1)
    a2 = np.concatenate((np.ones((1, m)), sigmoid(z2)), axis=0)  # <- calculate hidden layer and add bias unit before calculating output layer


    z3 = np.dot(Theta2, a2)
    a3 = sigmoid(z3)

    p+= np.argmax(a3, axis=0)  # <- # return index (aka class) of maximum h (we are comparing across rows; prediction for each EXAMPLE)


    # ============ can also try this: ================ 
    # given function has parameters 
        # num_layers : int
            # number of 
        # params : dict
            # dictionary of key-value pairs containing arrays of weights


    # X = np.concatenate((np.ones((m, 1)), X), axis=1)
    # a = X.transpose()
    # keys = list(params.keys())  # dict.keys() method returns dictionary keys
    
    # for i in range(num_layers-1):
        # z = np.dot(params[keys[i]], a)
        # a = sigmoid(z)

        #if i != num_layers-2:
            # a = np.concatenate((np.ones(1,m), a), axis=0)  # <- only adds bias unit if not output layer


    #p += np.argmax(a, axis=0)
    # =============================================== 

    return p
```
