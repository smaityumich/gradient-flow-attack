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



# Casing to tensor 
y_train, y_test = y_train.astype('int32'), y_test.astype('int32')
x_unprotected_train, x_unprotected_test = tf.cast(x_unprotected_train, dtype = tf.float32), tf.cast(x_unprotected_test, dtype = tf.float32)
y_train, y_test = tf.one_hot(y_train, 2), tf.one_hot(y_test, 2)
unprotected_directions = tf.cast(unprotected_directions, dtype = tf.float32)

init_graph = utils.ClassifierGraph(50, 2)
graph = cl.Classifier(init_graph, x_unprotected_train, y_train, x_unprotected_test, y_test, num_steps = 10000) # use for unfair algo
#graph = cl.Classifier(init_graph, tf.matmul(x_unprotected_train, unprotected_directions), 
#                        y_train, tf.matmul(x_unprotected_test, unprotected_directions), y_test, num_steps = 10000) # for fair algo



def distance_ratio(data_point, regularizer = 1e0, learning_rate = 1e-3, num_steps = 200):
    x, y = data_point
    x = tf.reshape(x, (1, -1))
    y = tf.reshape(y, (1, -1))
    x_start = x
    x_base = x
    x_fair = x
    for _ in range(num_steps):
        with tf.GradientTape() as g:
            g.watch(x_base)
            prob_base = graph(x_base)
            prob_start = graph(x_start)
            perturb = x - x_start
            loss = (utils.EntropyLoss(y, prob_base)-utils.EntropyLoss(y, prob_start))/(tf.norm(perturb)+1)

        gradient = g.gradient(loss, x_base)
        x_base = x_base + learning_rate * gradient 

    
    for _ in range(num_steps):
        with tf.GradientTape() as g:
            g.watch(x_fair)
            prob_fair = graph(x_fair)
            prob_start = graph(x_start)
            perturb = tf.matmul(x - x_start, unprotected_directions)
            loss = (utils.EntropyLoss(y, prob_fair)-utils.EntropyLoss(y, prob_start))/(tf.norm(perturb)+1)

        gradient = g.gradient(loss, x_fair)
        x_fair = x_fair + learning_rate * gradient 

    prob_fair = graph(x_fair)
    prob_base = graph(x_base)
    prob_start = graph(x_start)

    lf_fair = utils.kl(prob_fair, prob_start)/ (tf.norm(tf.matmul(x_fair-x_start, unprotected_directions)) + 1)
    lf_base = utils.kl(prob_base, prob_start)/ (tf.norm(tf.matmul(x_base-x_start, unprotected_directions)) + 1)
    lb_fair = utils.kl(prob_fair, prob_start)/ (tf.norm(x_fair-x_start) + 1)
    lb_base = utils.kl(prob_base, prob_start)/ (tf.norm(x_base-x_start) + 1)

    


    return (lf_fair/lb_fair-lb_base/lf_base).numpy()




cpus = mp.cpu_count()
print(f'Number of cpus : {cpus}')
start_time = time.time()
with mp.Pool(cpus) as pool:
    test_distance_ratios = pool.map(distance_ratio, zip(x_unprotected_test, y_test))
end_time = time.time()
test_distance_ratios = np.array(test_distance_ratios)


expt = '_1'
filename = f'output/test_distance_ratios{expt}.npy'


np.save(filename, test_distance_ratios)

#input = tf.keras.Input(shape=(39,), dtype='float32', name='input')
#output = graph.call(input)
#model = tf.keras.Model(inputs=input, outputs=output)
#tf.keras.utils.plot_model(model, to_file = imagename, show_shapes=True)


