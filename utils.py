import tensorflow as tf
from tensorflow import keras
import numpy as np

class ClassifierGraph(keras.Model):

    def __init__(self, n_hidden1, num_classes):
        super(ClassifierGraph, self).__init__()
        self.layer1 = keras.layers.Dense(n_hidden1, activation = tf.nn.relu)
        self.out = keras.layers.Dense(num_classes, activation = tf.nn.softmax)

    def call(self, x, training = False):
        x = self.layer1(x)
        x = self.out(x)
        if training:
            x = tf.nn.softmax(x)
        return x


def EntropyLoss(y, prob):
    return -2*tf.reduce_mean(tf.math.multiply(y, tf.math.log(prob)))


def _accuracy(y, ypred):
    acc = tf.cast(tf.equal(y, ypred), dtype = tf.float32)
    return tf.reduce_mean(acc)










