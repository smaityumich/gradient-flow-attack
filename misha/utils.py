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

def kl(p, q)
    return tf.reduce_sum(q * tf.math.log(q/p))


def gram_schmidt(X):
    """
    Implements Gram-Schmidt orthogonalization.

    Parameters
    ----------
    X : an n x k array with linearly independent columns

    Returns
    -------
    U : an n x k array with orthonormal columns

    """

    # Set up
    n, k = X.shape
    U = np.empty((n, k))
    I = np.eye(n)

    # The first col of U is just the normalized first col of X
    v1 = X[:,0]
    U[:, 0] = v1 / np.sqrt(np.sum(v1 * v1))

    for i in range(1, k):
        # Set up
        b = X[:, i]       # The vector we're going to project
        Z = X[:, 0:i]     # First i-1 columns of X

        # Project onto the orthogonal complement of the col span of Z
        M = I - Z @ np.linalg.inv(Z.T @ Z) @ Z.T
        u = M @ b

        # Normalize
        U[:, i] = u / np.sqrt(np.sum(u * u))

    return U

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









