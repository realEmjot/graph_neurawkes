import tensorflow as tf
from cont_lstm import ContLSTMCell, ContLSTMTrainer
from intensity import Intensity


cell = ContLSTMCell(10, 4)
trainer = ContLSTMTrainer(cell)
intensity = Intensity(10, 3)

T = 5.0
N = 10
######################
x_seq = tf.constant([1, 0, 2, 1])
t_seq = tf.constant([1.1, 2.3, 3.5, 4.7], dtype=tf.float32)
######################
inter_t_seq = tf.random.uniform([N], 0., T)

h_base, h_inter = trainer.train(x_seq, t_seq, inter_t_seq)

_, pos_lhd = tf.map_fn(
    fn=lambda xh: (0, intensity.get_intensity_for_type(*xh)),
    elems=(x_seq, h_base)
)
pos_lhd = tf.reduce_sum(tf.log(pos_lhd))

neg_lhd = intensity.get_all_intesities_batched(h_inter)
neg_lhd = tf.reduce_sum(neg_lhd, axis=1)
neg_lhd = T * tf.reduce_sum(neg_lhd) / N

lhd = pos_lhd - neg_lhd

optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(-lhd)

with tf.Session() as sess:
    sess.run(tf.initializers.global_variables())
    sess.run(train_step)
