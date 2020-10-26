"""In this file we store the modified implementations of the original methods used in our project"""

import numpy as np
from implementations import *


#################################################################
# Logistic regression using gradient descent - Modified version #
#################################################################

def calculate_loss_MODIFIED(y, tx, w):
    """compute the loss: negative log likelihood."""
    loss = np.logaddexp(0,tx@w)
    loss = loss - y*(tx@w)
    loss = np.sum(loss)
    loss = loss/len(y)
    return loss

def calculate_gradient_MODIFIED(y, tx, w):
    """compute the gradient of loss."""
    grad = tx.T.dot(sigmoid(tx.dot(w)) - y)/len(y)
    return grad

def learning_by_gradient_descent_MODIFIED(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss_MODIFIED(y,tx,w)
    grad = calculate_gradient_MODIFIED(y,tx,w)
    w = w - gamma*grad
    
    return loss, w

def logistic_regression_gradient_descent_demo_MODIFIED(y, tx, w, _gamma, _max_iter):
    # init parameters
    max_iter = _max_iter
    gamma = _gamma
    threshold = 1e-6
    losses = []

    #start the logistic regression
    for iter in range(max_iter):
        #for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
        # get loss and update w.
        loss, w = learning_by_gradient_descent_MODIFIED(y, tx, w, gamma)
        losses.append(loss)
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=losses[-1]))
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return losses[-1], w



############################################################################
# Regularized logistic regression using gradient descent - Modified version#
############################################################################

def penalized_logistic_regression_MODIFIED(y, tx, w, lambda_):
    """return the loss, gradient"""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient, and Hessian: TODO
    # ***************************************************
    loss = calculate_loss_MODIFIED(y, tx, w) + lambda_*np.squeeze(w.T.dot(w))
    gradient = calculate_gradient_MODIFIED(y, tx, w) + 2*lambda_*w
    return loss, gradient

def learning_by_penalized_gradient_MODIFIED(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient: TODO
    # ***************************************************
    loss, gradient = penalized_logistic_regression_MODIFIED(y, tx, w, lambda_)
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # update w: TODO
    # ***************************************************
    w = w - gamma * gradient
    return loss, w

def logistic_regression_penalized_gradient_descent_demo_MODIFIED(y, tx, w, _gamma, _max_iter, _lambda):
    """Regularized logistic regression using gradient descent"""
    # init parameters
    max_iter = _max_iter
    gamma = _gamma
    lambda_ = _lambda
    threshold = 1e-8
    losses = []

    #start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient_MODIFIED(y, tx, w, gamma, lambda_)
        loss = calculate_loss_MODIFIED(y,tx,w)
        losses.append(loss)
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=losses[-1]))
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return losses, w

