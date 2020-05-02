import tensorflow as tf
from tensorflow import keras
import numpy as np

class ClassifierGraph(keras.Model):

    def __init__(self, n_hidden1, num_classes):
        super(ClassifierGraph, self).__init__()
        self.layer1 = keras.layers.Dense(n_hidden1, activation = tf.nn.relu, name = 'layer-1')
        self.out = keras.layers.Dense(num_classes, activation = tf.nn.softmax, name = 'output')

    def call(self, x, predict = False):
        x = self.layer1(x)
        x = self.out(x)
        return tf.cast(tf.argmax(x, axis = 1), dtype = tf.float32) if predict else x


def EntropyLoss(y, prob):
    return -2*tf.reduce_mean(tf.math.multiply(y, tf.math.log(prob)))


def _accuracy(y, ypred):
    acc = tf.cast(tf.equal(y, ypred), dtype = tf.float32)
    return tf.reduce_mean(acc)










