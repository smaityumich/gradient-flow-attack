import numpy as np
import tensorflow as tf
import utils
import itertools
import multiprocessing as mp
from functools import partial

def linear_classifier(theta):
    theta = tf.cast(theta, dtype = tf.float32)
    theta = tf.reshape(theta, (-1, 1))
    def classifier(x):
        logits = x @ theta
        probs = 1 / (1 + tf.exp(-logits))
        return tf.concat([1-probs, probs], axis= 1)
    return classifier


def fair_metric_fn(theta):
    theta = tf.cast(theta, dtype = tf.float32)
    theta = tf.reshape(theta, (-1, 1))
    def fair_metric(x):
        return tf.norm(x @ theta, axis = 1)
    return fair_metric

def sample_perturbation(x, y, classifier, fair_metric, regularizer = 5, learning_rate = 2e-2, num_steps = 200):
    x = tf.reshape(x, (1, -1))
    y = tf.reshape(y, (1, -1))
    x_start = x
    x_fair = x
    #x += tf.cast(np.random.normal(size=(1, 39)), dtype = tf.float32)*1e-9
    for _ in range(num_steps):
        with tf.GradientTape() as g:
            g.watch(x_fair)
            prob = classifier(x_fair)
            loss = utils.EntropyLoss(y, prob)  - regularizer * tf.reduce_sum(fair_metric(x_fair - x_start)**2)

        gradient = g.gradient(loss, x_fair)
        x_fair = x_fair + learning_rate * gradient#utils.protected_direction(gradient, sensetive_directions)

    ratio = utils.EntropyLoss(y, classifier(x_fair)) / utils.EntropyLoss(y, classifier(x_start))
    return ratio.numpy()


def sample_perturbation_l2_base(x, y, classifier, fair_metric, regularizer = 5, learning_rate = 2e-2, num_steps = 200):
    x = tf.reshape(x, (1, -1))
    y = tf.reshape(y, (1, -1))
    x_start = x
    x_fair = x
    x_base = x
    #x += tf.cast(np.random.normal(size=(1, 39)), dtype = tf.float32)*1e-9
    for _ in range(num_steps):
        with tf.GradientTape() as g:
            g.watch(x_fair)
            prob = classifier(x_fair)
            loss = utils.EntropyLoss(y, prob)  - regularizer * tf.reduce_sum(fair_metric(x_fair - x_start)**2)

        gradient = g.gradient(loss, x_fair)
        x_fair = x_fair + learning_rate * gradient#utils.protected_direction(gradient, sensetive_directions)

    for _ in range(num_steps):
        with tf.GradientTape() as g:
            g.watch(x_base)
            prob = classifier(x_base)
            perturb = x_base-x_start
            loss = utils.EntropyLoss(y, prob)  - regularizer * tf.norm(perturb)**2

        gradient = g.gradient(loss, x_base)
        x_base = x_base + learning_rate * gradient

    

    ratio = utils.EntropyLoss(y, classifier(x_fair)) / utils.EntropyLoss(y, classifier(x_base))
    
    return ratio.numpy()



def mean_ratio(theta, fair_direction, regularizer = 1, learning_rate = 5e-2, num_steps = 200):
    x, y = np.load('data/x.npy'), np.load('data/y.npy')
    x, y = tf.cast(x, dtype = tf.float32), y.astype('int32')
    y = tf.one_hot(y, 2)

    cl = linear_classifier(theta)
    while np.linalg.norm(fair_direction) != 1:
        fair_direction = fair_direction / np.linalg.norm(fair_direction)
    fair_metric = fair_metric_fn(fair_direction)
    ratios = []
    for u, v in zip(x, y):
        r = sample_perturbation(u, v, classifier  = cl, fair_metric = fair_metric, regularizer=regularizer,\
             learning_rate=learning_rate, num_steps=num_steps)

        ratios.append(r)
    print(f'Done for mean ratio of {theta}')
    return np.mean(ratios)


def mean_ratio_l2_base(theta, fair_direction, regularizer = 1, learning_rate = 5e-2, num_steps = 200):
    x, y = np.load('data/x.npy'), np.load('data/y.npy')
    x, y = tf.cast(x, dtype = tf.float32), y.astype('int32')
    y = tf.one_hot(y, 2)

    cl = linear_classifier(theta)
    while np.linalg.norm(fair_direction) != 1:
        fair_direction = fair_direction / np.linalg.norm(fair_direction)
    fair_metric = fair_metric_fn(fair_direction)
    ratios = []
    for u, v in zip(x, y):
        r = sample_perturbation_l2_base(u, v, classifier  = cl, fair_metric = fair_metric, regularizer=regularizer,\
             learning_rate=learning_rate, num_steps=num_steps)

        ratios.append(r)
    print(f'Done for mean ratio l2 base of {theta}')
    return np.mean(ratios)


theta1 = np.arange(0, 5, step = 0.2)
theta2 = np.arange(0, 5, step= 0.2)
thetas = itertools.product(theta1, theta2)
theta = [list(i) for i in thetas]

fair_direction = [0, 1]

cpus = mp.cpu_count()
print(f'Number of cpus {cpus}')
with mp.Pool(cpus) as pool:
    mean_ratio_theta = pool.map(partial(mean_ratio, fair_direction = fair_direction, regularizer = 1,\
        learning_rate = 5e-2, num_steps = 100), theta)

    mean_ratio_theta_l2_base = pool.map(partial(mean_ratio_l2_base, fair_direction = fair_direction, regularizer = 1,\
        learning_rate = 5e-2, num_steps = 100), theta)

np.save('data/mean_ratio_theta_fair.npy', np.array(mean_ratio_theta))
np.save('data/mean_ratio_theta_l2_base_fair.npy', np.array(mean_ratio_theta_l2_base))
