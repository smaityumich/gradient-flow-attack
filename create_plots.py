import numpy as np
import tensorflow as tf
from adult_modified import preprocess_adult_data
from sklearn import linear_model
import classifier as cl
import utils
import time
import multiprocessing as mp
import dill
import random
import matplotlib.pyplot as plt
import scipy

seed = 1
dataset_orig_train, dataset_orig_test = preprocess_adult_data(seed = seed)

x_unprotected_train, x_protected_train = dataset_orig_train.features[:, :39], dataset_orig_train.features[:, 39:]
x_unprotected_test, x_protected_test = dataset_orig_test.features[:, :39], dataset_orig_test.features[:, 39:]
y_train, y_test = dataset_orig_train.labels.reshape((-1,)), dataset_orig_test.labels.reshape((-1,))





## Running linear regression to get sensetive directions 

protected_regression = linear_model.LinearRegression(fit_intercept = False)
protected_regression.fit(x_unprotected_train, x_protected_train)
sensetive_directions = protected_regression.coef_

def projection_matrix(sensetive_directions):
    n, d = sensetive_directions.shape
    mx = np.identity(d)
    for vector in sensetive_directions:
        vector = vector/np.linalg.norm(vector, ord=2)
        vector = vector.reshape((-1,1))
        mx = mx - 0.99* vector @ vector.T
    return mx




unprotected_directions = projection_matrix(sensetive_directions)



# Casing to tensor 
y_train, y_test = y_train.astype('int32'), y_test.astype('int32')
x_unprotected_train, x_unprotected_test = tf.cast(x_unprotected_train, dtype = tf.float32), tf.cast(x_unprotected_test, dtype = tf.float32)
y_train, y_test = tf.one_hot(y_train, 2), tf.one_hot(y_test, 2)

init_graph = utils.ClassifierGraph(50, 2)
graph = cl.Classifier(init_graph, x_unprotected_train, y_train, x_unprotected_test, y_test, num_steps = 1000)

unprotected_directions = tf.cast(unprotected_directions, dtype = tf.float32)





filename = 'adversarial-points/perturbed_test_points1.npy'
histplot = 'adversarial-points/perturbed-mean-entropy-hist1.png'
qqplot = 'adversarial-points/perturbed-mean-entropy-qqplot1.png'


perturbed_test_samples =  np.load(filename)


def error(data):
    x, y = data
    x = tf.cast(x, dtype = tf.float32)
    return utils.EntropyLoss(y, graph(x))

perturbed_error = [error(data) for data in zip(perturbed_test_samples, y_test)]
perturbed_error = [x.numpy() for x in perturbed_error]


def perturb_mean(n = 9045):
    index = random.sample(range(n), 400)
    srswr_perturb_errors =[perturbed_error[i] for i in index]
    return np.mean(srswr_perturb_errors)

perturbed_means = [perturb_mean() for _ in range(5000)]
plt.hist(perturbed_means)
plt.savefig(histplot)


scipy.stats.probplot(perturbed_means, plot=plt)
plt.savefig(qqplot)
