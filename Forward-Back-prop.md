# Optimization
## a) Compute cost and gradient
![](http://mathurl.com/render.cgi?%24%24%20J%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Csum_%7Bk%3D1%7D%5E%7BK%7D%20%5Cleft%5B%20-%20y_k%5E%7B%28i%29%7D%20%5Clog%20%5Cleft%28%20%5Cleft%28%20h_%5Ctheta%20%5Cleft%28%20x%5E%7B%28i%29%7D%20%5Cright%29%20%5Cright%29_k%20%5Cright%29%20-%20%5Cleft%28%201%20-%20y_k%5E%7B%28i%29%7D%20%5Cright%29%20%5Clog%20%5Cleft%28%201%20-%20%5Cleft%28%20h_%5Ctheta%20%5Cleft%28%20x%5E%7B%28i%29%7D%20%5Cright%29%20%5Cright%29_k%20%5Cright%29%20%5Cright%5D%20+%20%5Cfrac%7B%5Clambda%7D%7B2%20m%7D%20%5Cleft%5B%20%5Csum_%7Bj%3D1%7D%5E%7B25%7D%20%5Csum_%7Bk%3D1%7D%5E%7B400%7D%20%5Cleft%28%20%5CTheta_%7Bj%2Ck%7D%5E%7B%281%29%7D%20%5Cright%29%5E2%20+%20%5Csum_%7Bj%3D1%7D%5E%7B10%7D%20%5Csum_%7Bk%3D1%7D%5E%7B25%7D%20%5Cleft%28%20%5CTheta_%7Bj%2Ck%7D%5E%7B%282%29%7D%20%5Cright%29%5E2%20%5Cright%5D%20%24%24%20%20%20%5Cnocache)  
![](http://chart.apis.google.com/chart?cht=tx&chl=$\delta^{(l=L)}=a^{(L)}-y^{(i)}$)  
![](http://mathurl.com/render.cgi?%0A%24D_%7Bij%7D%5E%7B%28l%29%7D%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%20%28%5CDelta_%7Bij%7D%5E%7B%28l%29%7D%29%20%5Cquad%20%5Ctext%7Bif%20%7D%20j%5Cneq0%24%20%20%20%20%20%20%0A%0A%24%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5CTheta_%7Bij%7D%5E%7B%28l%29%7D%7D%20J%28%5CTheta%29%20%3D%20D_%7Bij%7D%5E%7B%28l%29%7D%24%0A%5Cnocache)  
![](http://mathurl.com/render.cgi?%24%24%20%5Ctext%7Bif%20g%28z%29%20is%20the%20sigmoid%20function%7D%20%5Cquad%20g%27%28z%29%5Cvert%20_%7Bz%3Da%5E%7B%28l%29%7D%7D%20%3D%20a%5E%7B%28l%29%7D%20.*%20%281%20-%20a%5E%7B%28l%29%7D%29%24%24%5Cnocache)  
```
def sigmoidGradient(a):
    """
    Computes gradient/derivative of sigmoid function

    Parameters
    ----------
    a : float
        Number at which g'(z) is computed. 
        If a is an array g'(z) computed element-wise.

    Returns
    ----------
    g : float
        derivative of sigmoid function, g(z), at z = a.
    """
    g = np.multiply(a, (1-a))

    return g
```  
```
def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_=0.0):    # <- NOTE: no non-default args can follow a default arg; default arg must be at end
    """
    Calculate cost using forward propogation and 
    gradient of neural network using backpropogation. 
    Works for regularized and nonregularized

    Parameters
    ----------
    nn_params : array_like
        All weights unrolled into a vector
        nn_params = np.array([np.ravel(Theta1), np.ravel(Theta2)])

    input_layer_size : int
        The number of units in the input layer;
        if imgs are a x b dimensions input_layer_size = a*b

    hidden_layer_size : int
        The number of units in the hidden layer

    num_labels : int
        Number of classification labels.
        aka the number of units in the output layer

    X : array_like
        Input dataset. Matrix of shape (m, input_layer_size).
        Does not include bias units

    y : array_like
        Dataset labels. A vector of shape (m,).

    lambda_ : float, optional
        Regularization parameter.

    Returns
    ----------
    J : float
        Cost computed at current weight values

    grad : array_like
        "Unrolled" vector of gradients for Theta1 and Theta2
    """
    m = y.size
    J = 0
    grad = np.zeros((m, 1))

    # Calculate cost
        # re-roll weights given have been unrolled and saved in nn_params
        # also given hidden_layer_size is number of units in layer 2
    Theta1 = np.reshape(nn_params[:(hidden_layer_size * (input_layer_size + 1))], 
                        (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):], 
                        (num_labels, (hidden_layer_size + 1)))    # <- NOTE: output_layer_size = num_labels

        # forward propogation to calculate h
    a1 = np.concatenate((np.ones((m, 1)), X), axis=1)
    z2 = np.dot(a1, Theta1.transpose())

    a2 = np.concatenate((np.ones((m,1)), sigmoid(z2)), axis=1)
    z3 = np.dot(a2, Theta2.transpose())

    h = sigmoid(z3)
    
        # convert y to matrix of 0s and 1s
    y_matrix = np.zeros((m, num_labels))
    for row in range(0,m):                # <- cycle through each row in matrix and set column corresponding to label number to 1
        y_matrix[row, int(y[row])] = 1    # NOTE indexes are returned as lists but we want integer value


    # =====================================================
    # given Theta1, Theta2, ... ThetaN saved in dictionary
    # weights = {Theta1 : np.reshape(nn_params[:dim1], dim1),
    #            Theta2 : np.reshape(nn_params[dim1:(dim1 + dim2)], dim2),
    #            Theta3 : np.reshape(nn_params[(dim1 + dim2):(dim1 + dim2 + dim3)], dim3)
    #                                 .     .      .
    #            ThetaN : np.reshape(nn_params[(dim1 + dim2 + dim3 +...+ dimN-1):(dim1 + dim2 + dim3 +...+ dimN-1 + dimN)], dimN}

    # NOTE: here dim1 = (s_2, s_1 + 1); dim2 = (s_3, s_2 + 1); dim3 = (s_4, s_3 + 1) ; dimN = (s_(N+1), s_N + 1)) 
    # =====================================================

        # NOTE: we want to do operations at every example along every class group
    cost = np.add(np.multiply(y_matrix, np.log(h)), np.multiply((1-y_matrix), np.log(1-h)))

    regterm = (lambda_/(2*m)) * ((Theta1[:,1:] ** 2).sum() + (Theta2[:, 1:] ** 2).sum())  # <- NOTE: we are summing for each layer and excluding weights corresponding to bias terms
    
    # =====================================================
    # given Theta1, Theta2, ... ThetaN
    # regterm = (lambda_/(2*m)) * ((Theta1[:,1:] ** 2).sum() + (Theta2[:,1:] ** 2).sum() + ... + (ThetaN[:,1:] ** 2).sum())
    # =====================================================

    J+= cost.sum()
    J *= (-1/m)

    J += regterm

    # Calculate gradient
    Theta1_grad = np.zeros(Theta1.shape)    # <- notice .shape ATTRIBUTE returns tuple so we don't use double parenthesis here
    Theta2_grad = np.zeros(Theta2.shape)

        # calculate "errors"
    delta3 = h - y_matrix
    delta2 = np.multiply(np.dot(delta3, Theta2[:,1:]), sigmoidGradient(a2[:,1:]))

    Delta1 = np.dot(a1.transpose(), delta2).transpose()   #<- Delta(l) should have same dimensions as Theta(l) if we are to add them later
    Delta2 = np.dot(a2.transpose(), delta3).transpose()

        # calculate gradients
    Theta1_copy = Theta1.copy()     # <- to be safe, copy matrices since we will be overwriting them
    Theta2_copy = Theta2.copy()

    Theta1_copy[:,0] = 0
    Theta2_copy[:,0] = 0 

    Theta1_grad += (1/m) * (Delta1 + (lambda_ * Theta1_copy))
    Theta2_grad += (1/m) * (Delta2 + (lambda_ * Theta2_copy))

    grad = np.concatenate((Theta1_grad.ravel(), Theta2_grad.ravel()), axis=0)
    

    return J, grad
```  
## b) Random Initialization
![](http://mathurl.com/render.cgi?%24%5Cepsilon_%7Binit%7D%20%3D%20%5Cfrac%7B%5Csqrt%7B6%7D%7D%7B%5Csqrt%7Bs_%7Bin%7D%20+%20s_%7Bout%7D%7D%7D%24%20where%20%24s_%7Bin%7D%20%3D%20s_l%24%20and%20%24s_%7Bout%7D%20%3D%20s_%7Bl+1%7D%24%20are%20the%20number%20of%20units%20in%20the%20layers%20adjacent%20to%20%24%5CTheta%5E%7Bl%7D%24%20%20%0A%0AInitialize%20weights%20%24%5Cin%20%5B-%5Cepsilon%2C%20%5Cepsilon%5D%24%20%20%5Cnocache)  
```
import math

def randInitializeWeights(s_in, s_out):
    """
    Randomly initialize weights of one layer in a neural
    network to small values within [-epsilon, +epsilon).

    Parameters
    ----------
    s_in : int
        s_(l), the number of units in layer "l"

    s_out : int
        s_(l+1), the number of units in layer "l+1"

    Returns
    ----------
    init_param : array_like
        Theta_(l), matrix of initialized weights
        for layer "l". Shape is (s_(l+1), s_l)

    """
    # calculate epsilon for layer
    epsilon = math.sqrt(6)/math.sqrt(s_in + s_out)

    # generate array of small numbers in [0,1) using epsilon
    init_param = np.zeros((s_out, (s_in + 1)))
    init_param += np.random.uniform((-1*epsilon), epsilon, (s_out, (s_in + 1))) * (2*epsilon) - epsilon     # np.random.rand(a,b) generates array of shape (a,b) of random num in [0,1) 
                                                                                                            # np.random.uniform(a,b, (c, d)) generates array of shape (a,b) of random num in [c,d)

    return init_param

# Using the function
Theta1 = randInitializeWeights(...)
Theta2 = randInitializeWeights(...)
.              .                 .
ThetaN = randInitializeWeights(...)

init_nn_params = np.concatenate((Theta1.ravel(), Theta2.ravel(), ..., ThetaN.ravel()), axis=0)   # => "unravelled" vector of initialized params
```
## c) Advanced Optimization
```
from scipy import optimize

init_nn_params = np.concatenate((Theta1.ravel(), Theta2.ravel(), ..., ThetaN.ravel()), axis=0)
options = {'maxiter' : n}    #<- where n is max number of iterations

res = optimize.minimize(nnCostFunction, init_nn_params,
                        (input_layer_size, 
                        hidden_layer_size,
                        num_labels, X, y, lambda_=0.0)
                        method='TNC', jac=True,
                        options=options)
cost = res.fun
unrolledTheta = res.x

#===========reshaping ThetaN example=============
# if ThetaN is of shape (s_N+1, s_N +1)
ThetaN = np.reshape(unrolledTheta[:(s_N+1 * s_N +1] + 1), (s_N+1, s_N +1))
#================================================
```
<br>


# Gradient Checking
> This works with any costFunction()  
![](http://mathurl.com/render.cgi?%24%5Ctext%7Bif%20%7D%20%5Ctheta%3D%5B%5Ctheta_i%2C%20%5Ctheta_%7Bi+1%7D%2C%20...%2C%20%5Ctheta_n%5D%20%5Ctext%7B%20is%20unrolled%20vector%20of%20all%20neural%20network%20parameters%7D%24%0A%0A%0A%24%5Cqquad%20%5Ctext%7Bfor%20%7D%20i%3D%281%2C%202%2C%203%2C%20...%2C%20n%29%24%0A%0A%24%5Cqquad%20%5Cquad%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%5Ctheta_%7Bi%7D%7DJ%28%5Ctheta%29%20%5Capprox%20%5Cfrac%7BJ%28%28%5Ctheta_i+%5Cepsilon%29%2C%20...%2C%20%5Ctheta_n%29%20-%20J%28%28%5Ctheta_i-%5Cepsilon%29%2C%20...%2C%20%5Ctheta_n%29%7D%7B2%5Cepsilon%7D%24%0A%0A%0A%0A%0A%5Cnocache)
```
def gradientCheck(params, costFunction, args=(), epsilon=1e-4):
    """
    Check optimized parameters by comparing
    them to an estimate of gradient.
    
    Parameters
    ----------
    params : array_like
        Optimized parameters for machine
        learning algorithm. Vector,
        therefore neural network optimized
        weights must be "unrolled".
        
    costFunction : object
        Function that calculates both cost, 
        gradient (in that order) for
        machine learning algorthm.
    
    args : tuple
        Other arguments needed by
        costFunction, not including
        vector of optimized parameters.
    
    epsilon : float, optional
        Error added and subtracted from
        theta when gradient is approximated.
    
    Returns
    ----------
        Printed outputs. Prints numrical gradients
        (approximation of gradient) and
        analytical gradients side-by-side.
    """
    gradients = np.zeros((params.size, 2))

    for i in range(params.size):
        Theta_plus = params.copy()
        Theta_plus[i] = Theta_plus[i] + epsilon        # <- returns entire nn_params just with a different number in position i

        Theta_minus = params.copy()
        Theta_minus[i] = Theta_minus[i] - epsilon
        
        # calculate costs
        J_plus, _ = costFunction(Theta_plus, *args)    # <- "*" expands a list or tupel and each element is passed to each argument
        J_minus, _ = costFunction(Theta_minus, *args)
        
        #compute numerical gradient
        approx_grad = (J_plus - J_minus)/(2*epsilon)
        
        gradients[i,0] = approx_grad
    
    # Calculate analytical gradient (NOTE we are compaing partial deriv of cost NOT Theta)
    _, grad = costFunction(params, *args)
    gradients[:,1] = grad
           
    
    # Print optimized parameters and approximated parameters side-by-side
    print("(Numerical Gradient, Analytical Gradient)")
    print(gradients)
```
> For complex neural networks running this function is EXTREMELY costly. So it would make more sense to:   
> (a) write a function, checkGradients() that runs computNumericalGradient() on a simulated neural  
> network of much smaller size or  
> (b) work an if else statement into computNumericalGradient() which prints user-inputted  
> segment of the output if params.size() > 20
<br>

# Predict
