import numpy as np
import tensorflow as tf

def get_inputs(n_dimension, n):
    theta_param = np.random.randn(n_dimension, 1)

    W_inputs = np.random.randn(n, n_dimension, n_dimension)
    y_inputs = np.zeros([n, n_dimension, 1])
    for i in range(n):
        y_inputs[i] = np.dot(W_inputs[i], theta_param)

    return W_inputs, y_inputs

w,y = get_inputs(15, 10)
print(w.shape, y.shape)


tf.squeeze()