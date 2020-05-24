import numpy as np
import tensorflow as tf
from adult_modified import preprocess_adult_data
from sklearn import linear_model
import classifier as cl
import utils
import time
import multiprocessing as mp
import random
import matplotlib.pyplot as plt
import scipy
plt.ioff()


seed = 1
dataset_orig_train, dataset_orig_test = preprocess_adult_data(seed = seed)

x_unprotected_train, x_protected_train = dataset_orig_train.features[:, :39], dataset_orig_train.features[:, 39:]
x_unprotected_test, x_protected_test = dataset_orig_test.features[:, :39], dataset_orig_test.features[:, 39:]
y_train, y_test = dataset_orig_train.labels.reshape((-1,)), dataset_orig_test.labels.reshape((-1,))





## Running linear regression to get sensetive directions 

protected_regression = linear_model.LinearRegression(fit_intercept = False)
protected_regression.fit(x_unprotected_train, x_protected_train)
sensetive_directions = protected_regression.coef_







unprotected_directions = utils.projection_matrix(sensetive_directions)
inv_unprotected_directions = np.linalg.inv(utils.projection_matrix(sensetive_directions, 0.01))



# Casing to tensor 
y_train, y_test = y_train.astype('int32'), y_test.astype('int32')
x_unprotected_train, x_unprotected_test = tf.cast(x_unprotected_train, dtype = tf.float32), tf.cast(x_unprotected_test, dtype = tf.float32)
y_train, y_test = tf.one_hot(y_train, 2), tf.one_hot(y_test, 2)
unprotected_directions = tf.cast(unprotected_directions, dtype = tf.float32)
inv_unprotected_directions = tf.cast(inv_unprotected_directions, dtype = tf.float32)

init_graph = utils.ClassifierGraph(50, 2)
graph = cl.Classifier(init_graph, x_unprotected_train, y_train, x_unprotected_test, y_test, num_steps = 10000) # use for unfair algo
#graph = cl.Classifier(init_graph, tf.matmul(x_unprotected_train, unprotected_directions), 
#                        y_train, tf.matmul(x_unprotected_test, unprotected_directions), y_test, num_steps = 10000) # for fair algo



def lipschitz(data_point, learning_rate = 5e-4):
    x, y, num_steps = data_point
    x = tf.reshape(x, (1, -1))
    y = tf.reshape(y, (1, -1))
    x_start = x
    for _ in range(num_steps):
        with tf.GradientTape() as g:
            g.watch(x)
            prob = graph(x)
            loss = utils.EntropyLoss(y, prob)

        gradient = g.gradient(loss, x)
        x = x + learning_rate * tf.matmul(gradient, inv_unprotected_directions) #/ tf.linalg.norm(gradient, ord = np.inf)
    prob2 = graph(x)
    prob1 = graph(x_start)
    dx = tf.norm((tf.matmul(x-x_start, unprotected_directions)))
    dy = utils.kl(prob2, prob1)
    return (dy/dx).numpy()


def mean_lipschitz(num_steps):
    cpus = mp.cpu_count()
    steps = [num_steps for _ in y_test]
    with mp.Pool(cpus) as pool:
        perturbed_test_samples = pool.map(lipschitz, zip(x_unprotected_test, y_test,  steps))
    print(f'Done for steps {num_steps}')
    return np.mean(perturbed_test_samples)

steps = [20, 40, 80, 160, 320, 640, 1280]#[20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
mean_lipschitz_const = [mean_lipschitz(s) for s in steps]
np.save('output/mean-lipschitz.npy', np.array(mean_lipschitz_const))



init_graph = utils.ClassifierGraph(50, 2)
#graph = cl.Classifier(init_graph, x_unprotected_train, y_train, x_unprotected_test, y_test, num_steps = 10000) # use for unfair algo
graph = cl.Classifier(init_graph, tf.matmul(x_unprotected_train, unprotected_directions), 
                        y_train, tf.matmul(x_unprotected_test, unprotected_directions), y_test, num_steps = 10000) # for fair algo




#steps = [20, 40, 80, 160, 320, 640, 1280]#[20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
mean_lipschitz_const = [mean_lipschitz(s) for s in steps]
np.save('output/mean-lipschitz-fair.npy', np.array(mean_lipschitz_const))


