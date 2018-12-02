import tensorflow as tf
from my_neurawkes import Neurawkes


x_seq = tf.constant([1, 0, 2, 1])
t_seq = tf.constant([1.1, 2.3, 3.5, 4.7], dtype=tf.float32)

model = Neurawkes(10, 3)

with tf.Session() as sess:
    model.train(sess, x_seq, t_seq, 10, 5.0, 1000)