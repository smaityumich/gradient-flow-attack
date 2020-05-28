import numpy as np
import tensorflow as tf
from adult_modified import preprocess_adult_data
from sklearn import linear_model
import utils









def Classifier(graph, x_train, y_train,  num_steps = 10000, batch_size = 250, learning_rate = 1e-4):
    batch_data = []
    # Tensor slice for train data
    batch = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    batch = batch.repeat().shuffle(5000).batch(batch_size)
    batch_data.append(batch.take(num_steps))

    

    # Adam optimizer
    optimizer = tf.optimizers.Adam(learning_rate)

    def train_step(data_train_epoch, step):
        x, y = data_train_epoch
        with tf.GradientTape() as g:
            loss = utils.EntropyLoss(y, graph(x, predict = False))

        variables = graph.trainable_variables
        gradients = g.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
    
    for step, data in enumerate(batch_data, 1):
        batch_data_train = data
        train_step(batch_data_train, step)
        if step % 200 == 0:
            print(f'Done step {step}\n')
    
    return graph

