import tensorflow as tf
from my_neurawkes import Neurawkes


x_seq = tf.constant([[1, 0, 2, 1], [0, 1, 1, 2]])
t_seq = tf.constant([[1.1, 2.3, 3.5, 4.7], [0.5, 1.0, 3.0, 4.2]], dtype=tf.float32)

N = 10
T = tf.constant([5.0, 4.5])

model = Neurawkes(10, 3)

with tf.Session() as sess:
    model.train(sess, x_seq, t_seq, N, T, 1000)
