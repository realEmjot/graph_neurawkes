import tensorflow as tf
import numpy as np

from cont_lstm import ContLSTMCell, ContLSTMTrainer

cell = ContLSTMCell(10, 4)

# x = tf.reshape(tf.constant([0., 1., 0.], dtype=tf.float32), [-1, 1])
# t_init = tf.constant(2.5, dtype=tf.float32)
# t = tf.constant(3.2, dtype=tf.float32)
# c_init = tf.zeros([10], dtype=tf.float32)
# c_base_init = tf.zeros([10], dtype=tf.float32)
# h_init = tf.zeros([10], dtype=tf.float32)

# res = cell(x, t_init, t, c_init, c_base_init, h_init)

trainer = ContLSTMTrainer(cell)

x_seq = tf.constant([
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=tf.float32)
t_seq = tf.constant([1.2, 2.3, 3.4], dtype=tf.float32)
inter_t_seq = tf.constant([1.8, 3.7], dtype=tf.float32)

outputs = trainer.train(x_seq, t_seq, inter_t_seq)
#print(outputs)

# writer = tf.summary.FileWriter('.')
# writer.add_graph(tf.get_default_graph())
# writer.close()

with tf.Session() as sess:
    sess.run(tf.initializers.global_variables())
    print(sess.run(outputs))