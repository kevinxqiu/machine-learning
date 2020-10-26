"""
Applying our best model to get a prediction
This prediction is the one that will be submitted into AICrowd
"""
import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from proj1_helpers import *
from impl_proj1 import *

#Loading data
DATA_TRAIN_PATH = 'data/train.csv' #data path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
DATA_TEST_PATH = 'data/test.csv'
y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#Feature cleaning
tX_rem = dealing_with_missing_data_mean(tX)
tX_rem = np.delete(tX_rem, [0, 3, 6, 7, 17, 18], axis=1)
tX_test_rem = dealing_with_missing_data_mean(tX_test)
tX_test_rem = np.delete(tX_test_rem, [0, 3, 6, 7, 17, 18], axis=1)

#Data standardization and parameters initialization
data_mean = tX_rem.mean(axis=0)
data_std = tX_rem.std(axis=0)
tX_std = standardize(tX_rem,data_mean,data_std)
tX_test_std = standardize(tX_test_rem,data_mean,data_std)
lambda_ = 0.026827
degree = 9

#Building model with ridge regression
tX_std_build = build_poly(tX_std,degree)
w, _ = ridge_regression(y, tX_std_build, lambda_)

#Tags prediction
tX_test_rem_build = build_poly(tX_test_rem,9)
y_pred = predict_labels(w, tX_test_rem_build)

#CSV generation
OUTPUT_PATH = 'sample-submission.csv'
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)