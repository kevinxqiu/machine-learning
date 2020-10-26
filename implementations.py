"""Mandatory methods for project 1"""

import numpy as np

############################################
# Linear regression using gradient descent #
############################################

def compute_loss_mse(y, tx, w):
    """Computes the loss using mse loss function"""
    e=y-tx@w
    e=e**2
    loss=1/2*np.mean(e)
    return loss

def compute_gradient_mse(y, tx, w):
    """Compute the gradient of the mse loss function"""
    N = len(y)
    e = y-tx@w
    grad = (-1/N) * np.transpose(tx) @ e
    return grad

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    loss = 0
    w = initial_w
    for n_iter in range(max_iters):
        #Gradient computation
        grad = compute_gradient_mse(y, tx, w)
        #w update by gradient
        w = w - gamma*grad
    
        loss = compute_loss_mse(y,tx,w)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss



#######################################################
# Linear regression using stochastic gradient descent #
#######################################################

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Divides the data in batches, shuffling data or not""" 
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    N = len(y)
    e = y-tx@w
    grad = (-1/N) * np.transpose(tx) @ e
    return grad


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        #Gradient and loss computation
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad = compute_stoch_gradient(y_batch,tx_batch,w)
            loss = compute_loss_mse(y_batch,tx_batch,w)
            #w update by gradient
            w = w - gamma*grad
            # store w and loss
            ws.append(w)
            losses.append(loss)

        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    return ws[-1], losses[-1]


###################################################
# Least squares regression using normal equations #
###################################################

def least_squares(y, tx):
    """Calculate the least squares solution."""
    left = tx.T.dot(tx)
    right = tx.T.dot(y)
    w = np.linalg.solve(left, right)
    loss = compute_loss_mse(y,tx,w)
    return w , loss



###########################################
# Ridge regression using normal equations #
###########################################

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = len(y)
    aI = 2 * N * lambda_ * np.eye(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    weight = np.linalg.solve(a, b)
    loss = compute_loss_mse(y,tx,weight)
    return weight, loss



#####################################################
# Logistic regression using gradient descent or SGD #
#####################################################

def sigmoid(t):
    """apply the sigmoid function on t."""

    return 1 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    loss = np.logaddexp(0,tx@w)
    loss = loss - y*(tx@w)
    loss = np.sum(loss)
    return loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    grad = tx.T.dot(sigmoid(tx.dot(w)) - y)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y,tx,w)
    grad = calculate_gradient(y,tx,w)
    w = w - gamma*grad
    
    return loss, w

def logistic_regression(y, tx , initial_w, max_iters, gamma):
    # init parameters
    max_iter = 10000
    threshold = 1e-8
    gamma = 0.01
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)),tx]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return losses[-1], w



##########################################################
# Regularized logistic regression using gradient descent #
##########################################################


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient, and Hessian: TODO
    # ***************************************************
    loss = calculate_loss(y, tx, w) + lambda_*np.squeeze(w.T.dot(w))
    gradient = calculate_gradient(y, tx, w) + 2*lambda_*w
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient: TODO
    # ***************************************************
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # update w: TODO
    # ***************************************************
    w = w - gamma * gradient
    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma): # 
    """Regularized logistic regression using gradient descent"""
    # init parameters
    max_iter = max_iter
    gamma = gamma
    lambda_ = lambda_
    threshold = 1e-8
    losses = []

    #start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        loss = calculate_loss(y,tx,w)
        losses.append(loss)
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=losses[-1]))
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return losses, w

