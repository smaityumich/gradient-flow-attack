import json
import tensorflow as tf

with open('data.txt', 'r') as f:
    data = json.load(f)

def model(x):
    n, _ = x.shape
    prob = tf.zeros([n, 1], dtype = tf.float32)
    for coef, intercept, weight in zip(data['coefs'], data['intercepts'], data['ens_weights']):
        coef = tf.cast(coef, dtype = tf.float32)
        coef = tf.reshape(coef, [-1, 1])
        model_logit = x @ coef + tf.cast(intercept, dtype = tf.float32)
        model_prob = tf.exp(model_logit) / (1 + tf.exp(model_logit))
        prob += model_prob * tf.cast(weight, dtype = tf.float32)

    return tf.concat([1-prob, prob], axis = 1)