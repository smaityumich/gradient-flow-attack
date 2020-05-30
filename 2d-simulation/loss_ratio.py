import numpy as np
import tensorflow as tf
import utils
import itertools
import multiprocessing as mp
from functools import partial
import sys
import random

def linear_classifier(theta):
    theta = np.array(theta)
    #theta = theta/(np.linalg.norm(theta) + 1e-16)
    def classifier(x):
        logits = np.sum(x * theta)
        if logits < 0:
            logits = np.log(np.exp(logits) + 1e-16)
        prob = 1 / (1 + np.exp(-logits))
        return prob
    return classifier


def fair_metric_fn(theta):
    
    theta = tf.cast(theta, dtype = tf.float32)
    #theta = tf.reshape(theta, (-1, 1))
    def fair_metric(x):
        return tf.norm(x @ theta, axis = 1)
    return fair_metric

def get_gradient(x, x_start, y,  theta, classifier, fair_direction, regularizer):
    prob = classifier(x)
    scalar = - 2 * regularizer * np.sum(fair_direction * (x - x_start))
    return (prob - y) * theta + scalar * fair_direction

def sample_perturbation(data, theta, fair_direction, regularizer = 5, learning_rate = 2e-2, num_steps = 200):
    x, y = data
    x_start = x
    x_fair = x
    classifier = linear_classifier(theta)
    fair_direction = np.array(fair_direction)
    theta = np.array(theta)
    #x += tf.cast(np.random.normal(size=(1, 39)), dtype = tf.float32)*1e-9
    for _ in range(num_steps):
        gradient = get_gradient(x_fair, x_start, y, theta,  classifier, fair_direction, regularizer)
        x_fair = x_fair + learning_rate * gradient

    ratio = utils.entropy(y, classifier(x_fair)) / utils.entropy(y, classifier(x_start))
    return ratio


def sample_perturbation_l2_base(data, theta, fair_direction, regularizer = 5, learning_rate = 2e-2, num_steps = 200):
    x, y = data
    classifier = linear_classifier(theta)
    fair_metric = fair_metric_fn(fair_direction)
    x = tf.reshape(x, (1, -1))
    y = tf.reshape(y, (1, -1))
    x_start = x
    x_fair = x
    x_base = x
    gradient0 = x_start
    #x += tf.cast(np.random.normal(size=(1, 39)), dtype = tf.float32)*1e-9
    for i in range(num_steps):
        with tf.GradientTape() as g:
            g.watch(x_fair)
            prob = classifier(x_fair)
            loss = utils.EntropyLoss(y, prob)  - regularizer * tf.reduce_sum(fair_metric(x_fair - x_start)**2)

        if i ==0:
            gradient = g.gradient(loss, x_fair)
        else:
            gradient0, gradient = gradient, g.gradient(loss, x_fair)
        
        if not tf.reduce_all(tf.math.is_finite(loss)):
            x_fair = x_fair - learning_rate * gradient0
            break
        else:
            x_fair = x_fair + learning_rate * gradient

    for _ in range(num_steps):
        with tf.GradientTape() as g:
            g.watch(x_base)
            prob = classifier(x_base)
            perturb = x_base-x_start
            loss = utils.EntropyLoss(y, prob)  - regularizer * tf.norm(perturb)**2

        gradient = g.gradient(loss, x_base)
        
        if not tf.reduce_all(tf.math.is_finite(loss)):
            x_base = x_base0
            break
        else:
            x_base0, x_base = x_base, x_base + learning_rate * gradient

    

    ratio = utils.EntropyLoss(y, classifier(x_fair)) / utils.EntropyLoss(y, classifier(x_base))
    
    return ratio.numpy()

def ratio_mean(x, sub_sample = 100):
    n = x.shape[0]
    index = np.random.randint(n, size = (100,))
    srswr_ratio=[x[i] for i in index]
    return np.mean(srswr_ratio)

def upper_ci(x, sub_sample = 100):
    sample_size = 400
    ratio_means = [ratio_mean(x, sub_sample=sub_sample) for _ in range(sample_size)]
    ratio_means = np.array(ratio_means)
    return np.mean(ratio_means) - 1.96 * np.std(ratio_means)/np.sqrt(sample_size)




def mean_ratio(theta, fair_direction, regularizer = 1, learning_rate = 5e-2, num_steps = 200):
    x, y = np.load('data/x.npy'), np.load('data/y.npy')
    
    
    cpus = mp.cpu_count()
    with mp.Pool(cpus) as pool:
        ratios = pool.map(partial(sample_perturbation, theta = theta, fair_direction = fair_direction,\
            regularizer = regularizer, learning_rate = learning_rate, num_steps = num_steps), zip(x, y))
    
    ratios = np.array(ratios)
    ratios = ratios[np.isfinite(ratios)]
    n = ratios.shape[0]
    mean = np.mean(ratios)
    std = np.std(ratios)
    print(f'Done for mean ratio of {theta} with mean, std, n {mean} {std} {n}')
    return mean - 1.96 * std/np.sqrt(n)#np.mean(ratios)


# def mean_ratio_l2_base(theta, fair_direction, regularizer = 1, learning_rate = 5e-2, num_steps = 200):
#     x, y = np.load('data/x.npy'), np.load('data/y.npy')
#     x, y = tf.cast(x, dtype = tf.float32), y.astype('int32')
#     y = tf.one_hot(y, 2)

#     cpus = mp.cpu_count()
    # with mp.Pool(cpus) as pool:
    #     ratios = pool.map(partial(sample_perturbation_l2_base, theta = theta, fair_direction = fair_direction,\
    #         regularizer = regularizer, learning_rate = learning_rate, num_steps = num_steps), zip(x, y))
    # print(f'Done for mean ratio l2 base of {theta}')
    # ratios = np.array(ratios)
    # ratios = ratios[np.isfinite(ratios)]
    # return np.mean(ratios)


theta1 = np.arange(0, 8.1, step = 0.5)
theta2 = np.arange(0, 5.1, step= 0.5)
thetas = itertools.product(theta1, theta2)
theta = [list(i) for i in thetas]

ang = int(float(sys.argv[1]))
angle = np.radians(ang)
c, s = np.cos(angle), np.sin(angle)
R = np.array(((c, -s), (s, c)))
fair_direction = np.array([[0], [1]])
fair_direction = R @ fair_direction
while np.linalg.norm(fair_direction) != 1:
    fair_direction = fair_direction/np.linalg.norm(fair_direction)


mean_ratio_theta = []
#mean_ratio_theta_l2_base = []
for t1 in theta1:
    mean_ratio_theta_row = []
    #mean_ratio_theta_l2_base_row = []
    for t2 in theta2:
        r = mean_ratio([t1, t2], fair_direction, regularizer= 5e2, learning_rate=1e-4, num_steps=1000)
        mean_ratio_theta_row.append(r)
        #r = mean_ratio_l2_base([t1, t2], fair_direction, regularizer= 2, learning_rate=2e-2, num_steps=100)
        #mean_ratio_theta_l2_base_row.append(r)
    mean_ratio_theta.append(mean_ratio_theta_row)
    #mean_ratio_theta_l2_base.append(mean_ratio_theta_l2_base_row)




np.save(f'data/test_stat_{ang}.npy', np.array(mean_ratio_theta))
#np.save(f'data/mean_ratio_l2_{ang}.npy', np.array(mean_ratio_theta_l2_base))
