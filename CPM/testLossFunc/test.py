import numpy as np
import tensorflow as tf

a = tf.TensorArray([[1, 2, 3], [3, 4, 4], [7, 8, 9]])
b = tf.TensorArray([[5, 6, 6], [7, 8, 8], [6, 7, 8]])
c = tf.stack([a, b], axis = 2)
d = tf.TensorArray([[1, 2, 3], [3, 4, 4], [6, 7, 8]])
e = tf.stack([c, tf.extend_dims(d)])
print(a.shape)
print(b.shape)
print(c[:, :, 0])
print(d.shape)
print(e.shape)
