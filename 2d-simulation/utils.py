import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy

class ClassifierGraph(keras.Model):

    def __init__(self, n_hidden1, num_classes):
        super(ClassifierGraph, self).__init__()
        self.layer1 = keras.layers.Dense(n_hidden1, activation = tf.nn.relu, name = 'layer-1')
        self.out = keras.layers.Dense(num_classes, activation = tf.nn.softmax, name = 'output')

    def call(self, x, predict = False):
        x = self.layer1(x)
        x = self.out(x)
        x, _ = tf.linalg.normalize(x, ord = 1, axis = 1)
        return tf.cast(tf.argmax(x, axis = 1), dtype = tf.float32) if predict else x


def EntropyLoss(y, prob):
    return -2*tf.reduce_mean(tf.math.multiply(y, tf.math.log(prob)))




def _accuracy(y, ypred):
    acc = tf.cast(tf.equal(y, ypred), dtype = tf.float32)
    return tf.reduce_mean(acc)




def gram_schmidt(x):
    y = []
    for i, u in enumerate(x):
        if i == 0:
            while np.linalg.norm(u) != 1:
                u = u/np.linalg.norm(u)
            y.append(u)
        else:
            while np.sum([np.absolute(np.sum(u * v)) for v in y]) > 1e-16:
                for v in y:
                    u -= np.sum(u * v)*v
                while  np.linalg.norm(u) != 1:
                    u = u/np.linalg.norm(u)
    return np.array(y)


def projection_matrix(sensetive_directions):
    orthogonal_sd = scipy.linalg.orth(sensetive_directions.T).T
    _, d = orthogonal_sd.shape

    mx = np.identity(d)
    for vector in sensetive_directions:
        vector = vector.reshape((-1,1))
        while np.linalg.norm(vector) != 1:
            vector = vector/np.linalg.norm(vector)
        mx = mx - vector @ vector.T
    return mx

def projection_matrix2(sensetive_directions):
    orthogonal_sd = scipy.linalg.orth(sensetive_directions.T).T
    _, d = orthogonal_sd.shape

    mx = np.zeros((d, d))
    for vector in sensetive_directions:
        vector = vector.reshape((-1,1))
        while np.linalg.norm(vector) != 1:
            vector = vector/np.linalg.norm(vector)
        mx = mx + vector @ vector.T
    return mx



def protected_direction(x, sensetive_directions):
    x = x @ tf.linalg.matrix_transpose(sensetive_directions) @ sensetive_directions
    return x
def unprotected_direction(x, sensetive_directions):
    x = x - x @ tf.linalg.matrix_transpose(sensetive_directions) @ sensetive_directions
    return x


def entropy(y, p):
    a = p if y else 1-p
    return -np.log(a+1e-16)

