import numpy as np
import tensorflow as tf
from adult_modified import preprocess_adult_data
from sklearn import linear_model
import utils

batch_size = 200
seed = 1
dataset_orig_train, dataset_orig_test = preprocess_adult_data(seed = seed)

x_unprotected_train, x_protected_train = dataset_orig_train.features[:, :39], dataset_orig_train.features[:, 39:]
x_unprotected_test, x_protected_test = dataset_orig_test.features[:, :39], dataset_orig_test.features[:, 39:]
y_train, y_test = dataset_orig_train.labels, dataset_orig_test.labels

x_unprotected_train, x_unprotected_test = tf.cast(x_unprotected_train, dtype = tf.float32), tf.cast(x_unprotected_test, dtype = tf.float32)
y_train, y_test = tf.one_hot(y_train, 2), tf.one_hot(y_test, 2)


graph = utils.ClassifierGraph(100, 2)



def Classifier(graph, x_train, y_train, x_test, y_test, num_steps = 10000, batch_size = 250, learning_rate = 1e-4):
    batch_data = []
    # Tensor slice for train data
    batch = tf.data.Dataset.from_tensor_slices((x_unprotected_train, y_train))
    batch = batch.repeat().shuffle(5000).batch(batch_size)
    batch_data.append(batch.take(num_steps))

    # Tensor slice for test data
    batch = tf.data.Dataset.from_tensor_slices((x_unprotected_test, y_test))
    batch = batch.repeat().shuffle(5000).batch(batch_size)
    batch_data.append(batch.take(num_steps))

    # Adam optimizer
    optimizer = tf.optimizers.Adam(learning_rate)

    def train_step(data_train_epoch, step):
        x, y = data_train_epoch
        with tf.GradientTape() as g:
            loss = utils.EntropyLoss(y, graph(x, training = True))

        variables = graph.trainable_variable
        gradients = g.gradients(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
    
    for step, data in enumerate(zip(*batch_data), 1):
        batch_data_train, _ = data
        train_step(batch_data_train, step)
    
    return graph

trained_graph = Classifier(graph, x_unprotected_train, y_train, x_unprotected_test, y_test)




## Run linear regression X_gender_train on y_gender_train to get the sensitive direction for the protected attribute gender

protected_regression = linear_model.LinearRegression(fit_intercept = False)
protected_regression.fit(x_unprotected_train, x_protected_train)
sensetive_directions = protected_regression.coef_
