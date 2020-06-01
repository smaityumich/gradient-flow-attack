import numpy as np
import tensorflow as tf
from adult_modified import preprocess_adult_data
from sklearn import linear_model
import utils
import time
import multiprocessing as mp
import random
import matplotlib.pyplot as plt
import scipy
plt.ioff()
import sys
#tf.compat.v1.enable_eager_execution()

def sample_perturbation(data_point, regularizer = 20, learning_rate = 3e-2, num_steps = 200):
    x, y = data_point
    x = tf.reshape(x, (1, -1))
    y = tf.reshape(y, (1, -1))
    x_start = x
    #x += tf.cast(np.random.normal(size=(1, 39)), dtype = tf.float32)*1e-9
    for _ in range(num_steps):
        with tf.GradientTape() as g:
            g.watch(x)
            prob = graph(x)
            perturb = utils.unprotected_direction(x-x_start, sensetive_directions)
            loss = utils.EntropyLoss(y, prob)  - regularizer / ((0 + 1) ** (2/3)) * tf.norm(perturb)**2

        gradient = g.gradient(loss, x)
        x = x + learning_rate * gradient#utils.protected_direction(gradient, sensetive_directions)

    return_loss = utils.EntropyLoss(y, graph(x)) / utils.EntropyLoss(y, graph(x_start))
    
    return return_loss.numpy()

if __name__ == '__main__':

    start, end = int(float(sys.argv[1])), int(float(sys.argv[2]))
    seed_data = int(float(sys.argv[3]))
    seed_model = int(float(sys.argv[4]))
    lr = float(sys.argv[5])
    dataset_orig_train, dataset_orig_test = preprocess_adult_data(seed = seed_data)

    x_unprotected_train, x_protected_train = dataset_orig_train.features[:, :39], dataset_orig_train.features[:, 39:]
    x_unprotected_test, x_protected_test = dataset_orig_test.features[:, :39], dataset_orig_test.features[:, 39:]
    y_train, y_test = dataset_orig_train.labels.reshape((-1,)), dataset_orig_test.labels.reshape((-1,))





## Running linear regression to get sensetive directions 

#protected_regression = linear_model.LinearRegression(fit_intercept = False)
#protected_regression.fit(x_unprotected_train, x_protected_train)
#sensetive_directions = protected_regression.coef_



    sensetive_directions = []
    protected_regression = linear_model.LogisticRegression(fit_intercept = True)
    protected_regression.fit(x_unprotected_test, x_protected_test[:, 0])
    sensetive_directions.append(protected_regression.coef_.reshape((-1,)))
    protected_regression.fit(x_unprotected_test, x_protected_test[:, 1])
    sensetive_directions.append(protected_regression.coef_.reshape((-1,)))
    sensetive_directions = np.array(sensetive_directions)

    sensetive_directions = scipy.linalg.orth(sensetive_directions.T).T
    for i, s in enumerate(sensetive_directions):
        while np.linalg.norm(s) != 1:
            s = s/ np.linalg.norm(s)
        sensetive_directions[i] = s





    unprotected_directions = utils.projection_matrix(sensetive_directions)
    protected_directions = utils.projection_matrix2(sensetive_directions)



# Casing to tensor 
    y_train, y_test = y_train.astype('int32'), y_test.astype('int32')
    x_unprotected_train, x_unprotected_test = tf.cast(x_unprotected_train, dtype = tf.float32), tf.cast(x_unprotected_test, dtype = tf.float32)
    y_train, y_test = tf.one_hot(y_train, 2), tf.one_hot(y_test, 2)
    unprotected_directions = tf.cast(unprotected_directions, dtype = tf.float32)
    protected_directions = tf.cast(protected_directions, dtype = tf.float32)
    sensetive_directions = tf.cast(sensetive_directions, dtype = tf.float32)



    graph = tf.keras.models.load_model(f'./baseline/graphs/graph_{seed_data}_{seed_model}')     



#cpus = mp.cpu_count()
#print(f'Number of cpus : {cpus}')

    perturbed_test_samples = []
    for data in zip(x_unprotected_test[start:end], y_test[start:end]):
        perturbed_test_samples.append(sample_perturbation(data, regularizer=50, learning_rate=lr, num_steps=200))
# with mp.Pool(cpus) as pool:
#     perturbed_test_samples = pool.map(sample_perturbation, zip(x_unprotected_test, y_test))
# end_time = time.time()
    perturbed_test_samples = np.array(perturbed_test_samples)



    filename = f'./baseline/outcome/perturbed_ratio_start_{start}_end_{end}_seed_{seed_data}_{seed_model}_lr_{lr}.npy'


    np.save(filename, perturbed_test_samples)

#input = tf.keras.Input(shape=(39,), dtype='float32', name='input')
#output = graph.call(input)
#model = tf.keras.Model(inputs=input, outputs=output)
#tf.keras.utils.plot_model(model, to_file = imagename, show_shapes=True)


