import numpy as np
import tensorflow as tf
from adult_modified import preprocess_adult_data, get_sensitive_directions_and_projection_matrix
from sklearn import linear_model

seed = 1
dataset_orig_train, dataset_orig_test = preprocess_adult_data(seed = seed)

## Run linear regression X_gender_train on y_gender_train to get the sensitive direction for the protected attribute gender
x_unprotected_train, x_protected_train = dataset_orig_train.features[:, :39], dataset_orig_train.features[:, 39:]
x_unprotected_test, x_protected_test = dataset_orig_test.features[:, :39], dataset_orig_test.features[:, 39:]
protected_regression = linear_model.LinearRegression(fit_intercept = False)
protected_regression.fit(x_unprotected_train, x_protected_train)
sensetive_directions = protected_regression.coef_
