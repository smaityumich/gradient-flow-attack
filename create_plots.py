import numpy as np
import tensorflow as tf
from adult_modified import preprocess_adult_data
from sklearn import linear_model
import classifier as cl
import utils
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
    _, d = sensetive_directions.shape
    mx = np.identity(d)
    for vector in sensetive_directions:
        vector = vector.reshape((-1,1))
        vector = vector/np.linalg.norm(vector, ord=2)
        mx = mx - 0.9999* vector @ vector.T
    return mx




unprotected_directions = projection_matrix(sensetive_directions)



# Casing to tensor 
y_train, y_test = y_train.astype('int32'), y_test.astype('int32')
x_unprotected_train, x_unprotected_test = tf.cast(x_unprotected_train, dtype = tf.float32), tf.cast(x_unprotected_test, dtype = tf.float32)
y_train, y_test = tf.one_hot(y_train, 2), tf.one_hot(y_test, 2)
unprotected_directions = tf.cast(unprotected_directions, dtype = tf.float32)

init_graph = utils.ClassifierGraph(50, 2)
#graph = cl.Classifier(init_graph, x_unprotected_train, y_train, x_unprotected_test, y_test, num_steps = 10000) # use for unfair algo
graph = cl.Classifier(init_graph, tf.matmul(x_unprotected_train, unprotected_directions), 
                        y_train, tf.matmul(x_unprotected_test, unprotected_directions), y_test, num_steps = 10000) # for fair algo


#probs = graph(x_unprotected_test)
probs = graph(tf.matmul(x_unprotected_test, unprotected_directions))
standard_error = utils.EntropyLoss(y_test, probs)



expt = 4
filename = f'adversarial-points/perturbed_test_points{expt}.npy'
l2_filename = f'adversarial-points/l2_perturbed_test_points{expt}.npy'
histplot = f'adversarial-points/perturbed-mean-entropy-hist{expt}.png'
qqplot = f'adversarial-points/perturbed-mean-entropy-qqplot{expt}.png'


perturbed_test_samples =  np.load(filename)
l2_perturbed_test_samples = np.load(l2_filename)


def error(data):
    global standard_error
    x_perturbed, x_l2_perturbed, y = data
    x_perturbed, x_l2_perturbed = tf.cast(x_perturbed, dtype = tf.float32), tf.cast(x_l2_perturbed, dtype = tf.float32)
    x_perturbed, x_l2_perturbed = tf.reshape(x_perturbed, (1, -1)), tf.reshape(x_l2_perturbed, (1, -1))
    y = tf.reshape(y, (1, -1))
    x_perturbed, x_l2_perturbed = tf.matmul(x_perturbed, unprotected_directions), tf.matmul(x_l2_perturbed, unprotected_directions) # for fair algo
    return utils.EntropyLoss(y, graph(x_perturbed)) - utils.EntropyLoss(y, graph(x_l2_perturbed))

perturbed_error = [error(data) for data in zip(perturbed_test_samples, l2_perturbed_test_samples, y_test)]
perturbed_error = [x.numpy() for x in perturbed_error]


def perturb_mean(n = 9045):
    index = random.sample(range(n), 400)
    srswr_perturb_errors =[perturbed_error[i] for i in index]
    return np.mean(srswr_perturb_errors)

perturbed_means = [perturb_mean() for _ in range(5000)]
plt.hist(perturbed_means)
plt.title(f'Histogram of mean loss of perturbed samples for expt {expt}')
plt.xticks(rotation = 35, fontsize = 'x-small')
plt.savefig(histplot)
plt.close()


scipy.stats.probplot(perturbed_means, plot=plt)
plt.title(f'Normal qq-plot of mean loss of perturbed samples for expt {expt}')
plt.savefig(qqplot)
