import numpy as np
import tensorflow as tf
from adult_modified import preprocess_adult_data
from sklearn import linear_model
import classifier as cl
import utils
import time
import multiprocessing as mp

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
print(unprotected_directions)


# Casing to tensor 
y_train, y_test = y_train.astype('int32'), y_test.astype('int32')
x_unprotected_train, x_unprotected_test = tf.cast(x_unprotected_train, dtype = tf.float32), tf.cast(x_unprotected_test, dtype = tf.float32)
y_train, y_test = tf.one_hot(y_train, 2), tf.one_hot(y_test, 2)

graph = utils.ClassifierGraph(50, 2)
graph = cl.Classifier(graph, x_unprotected_train, y_train, x_unprotected_test, y_test, num_steps = 1000)

unprotected_directions = tf.cast(unprotected_directions, dtype = tf.float32)

def sample_perturbation(x, y, regularizer = 1e-2, learning_rate = 1e-4, num_steps = 20):
    
    x = tf.reshape(x, (1, -1))
    y = tf.reshape(y, (1, -1))
    x_start = x
    for _ in range(num_steps):
        with tf.GradientTape() as g:
            g.watch(x)
            purturb = tf.linalg.matmul(x - x_start, unprotected_directions)
            prob = graph(x)
            loss = utils.EntropyLoss(y, prob) - regularizer * tf.reduce_sum(purturb**2)

        gradient = g.gradient(loss, x)
        x = x + learning_rate * gradient / tf.linalg.norm(gradient, ord = 2)
    return x

def perturbed_loss(x, y, regularizer = 1e-2, learning_rate = 1e-4, num_steps = 20):
    x_perturbed = sample_perturbation(x, y, regularizer, learning_rate, num_steps)
    return utils.EntropyLoss(y, graph(x_perturbed))


start_time = time.time()
for x, y in zip(x_unprotected_train[:100], y_train[:100]):
    perturbed_loss(x, y)
end_time = time.time()

print(f'Time taken {(end_time - start_time)/100}\n')
print(f'Total time taken {end_time-start_time}')
