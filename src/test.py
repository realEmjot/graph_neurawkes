import tensorflow as tf
import numpy as np

from cont_lstm import ContLSTMCell, ContLSTMTrainer
from intensity import Intensity


cell = ContLSTMCell(3, 4)
trainer = ContLSTMTrainer(cell)
intensity = Intensity(3, 4)

x_seq = tf.constant([
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=tf.float32)
t_seq = tf.constant([1.1, 2.3, 3.5], dtype=tf.float32)
inter_t_seq = tf.constant([2.3, 3.5], dtype=tf.float32)

h_base, h_inter = trainer.train(x_seq, t_seq, inter_t_seq)
base_ints = tf.map_fn(
    fn=lambda h: intensity(1, h),
    elems=h_base
)


with tf.Session() as sess:
    sess.run(tf.initializers.global_variables())
    print(sess.run(base_ints))