import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from implementations_modified import *
from proj1_helpers import *

#########################################
#           Data preprocessing          #
#########################################

def standardize(x,data_mean,data_std):
    norm_data = (x - data_mean) / data_std
    return norm_data

def change_tags(y):
    """Changes tags for logistic regression (-1 to 0)"""
    y = np.where(y==-1, 0, y)
    return y 

def undo_change_tags(y):
    """Restores tags (0 to -1)"""
    y = np.where(y==0, -1, y) 
    return y

def dealing_with_missing_data_mean(tX):
    """
    Drops those features related with jet_num value (they had lots of -999).
    The rest of -999 in other features are replaced by the mean.
    """
    tX_rem = np.delete(tX, [4,5,6,12,23,24,25,26,27,28], axis=1)
    Nan = np.nan
    tX_nan = np.where(tX_rem == -999, Nan, tX_rem)
    means = np.apply_along_axis(np.nanmean, 0, tX_nan)
    print(means)
    tX_ok = np.where(tX_rem == -999, means, tX_rem)
    return tX_ok


#########################################
#               Accuracy                #
#########################################

def check_accuracy(y_pred,y_real):
    """Returns model accuracy comparing prediction with real labels"""
    acc=np.sum(y_pred==y_real)/y_real.shape[0]
    return acc


#########################################
#         Feature augmentation          #
#########################################

def build_poly(x,degree):
    """Performs feature augmentation adding new degrees to our x matrix"""
    n=x.shape[0]
    d=x.shape[1]
    aux=np.ones((n,1))
    
    degrees=degree*np.ones(d).astype(int)
           
    for i, degree in enumerate(degrees):
        for j in range(degree):
            aux=np.c_[aux,x[:,i]**(j+1)]
    
    return aux



#########################################
#           Cross validation            #
#########################################

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, gamma,_max_iter,model,degree):
    """
    Performs cross validation taking into accout the model that we are using. Returns train and test accuracy
    """
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    initial_w = np.zeros((x_tr.shape[1]))

    if model=="GD":
        w, _ = least_squares_GD(y_tr, x_tr, initial_w, _max_iter, gamma)
        y_pred_train = predict_labels(w,x_tr)
        y_pred_test = predict_labels(w,x_te)
        acc_train = check_accuracy(y_pred_train,y_tr)
        acc_test = check_accuracy(y_pred_test,y_te)

    if model=="SGD":
        w, _ = least_squares_SGD(y_tr, x_tr, initial_w, _max_iter, gamma)
        y_pred_train = predict_labels(w,x_tr)
        y_pred_test = predict_labels(w,x_te)
        acc_train = check_accuracy(y_pred_train,y_tr)
        acc_test = check_accuracy(y_pred_test,y_te)

    if model=="LS":
        x_tr = build_poly(x_tr, degree)
        x_te = build_poly(x_te, degree)
        w, _ = least_squares(y_tr, x_tr)
        y_pred_train = predict_labels(w,x_tr)
        y_pred_test = predict_labels(w,x_te)
        acc_train = check_accuracy(y_pred_train,y_tr)
        acc_test = check_accuracy(y_pred_test,y_te)

    if model=="RR":
        x_tr = build_poly(x_tr, degree)
        x_te = build_poly(x_te, degree)
        w, _ = ridge_regression(y_tr, x_tr,lambda_)
        y_pred_train = predict_labels(w,x_tr)
        y_pred_test = predict_labels(w,x_te)
        acc_train = check_accuracy(y_pred_train,y_tr)
        acc_test = check_accuracy(y_pred_test,y_te)
    
    if model=="LR":
        _, w = logistic_regression_gradient_descent_demo_MODIFIED(y_tr, x_tr, initial_w, gamma, _max_iter)
        y_pred_train = predict_labels_logistic(w,x_tr)
        y_pred_test = predict_labels_logistic(w,x_te)
        acc_train = check_accuracy(y_pred_train,y_tr)
        acc_test = check_accuracy(y_pred_test,y_te)

    if model=="PLR":
        _, w = logistic_regression_penalized_gradient_descent_demo_MODIFIED(y_tr, x_tr, initial_w, gamma, _max_iter,lambda_)
        y_pred_train = predict_labels_logistic(w,x_tr)
        y_pred_test = predict_labels_logistic(w,x_te)
        acc_train = check_accuracy(y_pred_train,y_tr)
        acc_test = check_accuracy(y_pred_test,y_te)


    return acc_train, acc_test, w


def cross_validation_visualization(lambds, acc_tr, acc_te, degree,i):
    """visualization the curves of acc_train and acc_test."""

    #train_colors = ['b','green','red','orange']
    #test_colors = ['yellow','pink','tan','lime']
    train_colors = ['b','green','orange','tan']
    test_colors = ['r','pink','lime','yellow']
    plt.semilogx(lambds, acc_tr, marker=".", color=train_colors[i], label='Train degree '+str(degree))
    plt.semilogx(lambds, acc_te, marker=".", color=test_colors[i], label='Test degree '+str(degree))
    
    plt.xlabel("Lambda")
    plt.ylabel("Accuracy")
    plt.title("Cross Validation for Lambda")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig("cross_validation")

def cross_validation_demo(y, x, gamma, max_iters,model,degree,i):
    """
    This function executes cross validation. Creates the lambdas that we are going to compare and
    calls the visualization function. Returns the test and train accuracy for every lambda.
    """
    seed = 1
    k_fold = 5
    if model=="GD" or model=="SGD" or model=="LS" or model=="LR":
        lambdas = np.logspace(0, 1, 1)
    if model=="RR":
        lambdas = np.logspace(-4, -2, 8)
    if model=="PLR":
        lambdas = np.logspace(-5, -2, 6)
        
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    acc_tr = []
    acc_te = []
    w = []
    # ***************************************************
    # cross validation: 
    for lambda_ in lambdas:
        acc_tr_tmp = []
        acc_te_tmp = []
        for k in range(k_fold):
            acc_train, acc_test, w= cross_validation(y, x, k_indices, k, lambda_, gamma, max_iters,model,degree)
            acc_tr_tmp.append(acc_train)
            acc_te_tmp.append(acc_test)
        acc_tr.append(np.mean(acc_tr_tmp))
        acc_te.append(np.mean(acc_te_tmp))
    # ***************************************************   
    
    if model =="RR" or model=="PLR":
        cross_validation_visualization(lambdas, acc_tr, acc_te,degree,i)
    
    return acc_tr, acc_te, w
