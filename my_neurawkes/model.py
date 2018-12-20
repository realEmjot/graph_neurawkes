import tensorflow as tf
from tqdm import trange

from .cont_lstm import ContLSTMCell, ContLSTMTrainer
from .intensity import Intensity


class Neurawkes:
    def __init__(self, num_units, num_types):
        self._cell = ContLSTMCell(num_units, num_types + 1)
        self._trainer = ContLSTMTrainer(self._cell)
        self._intensity = Intensity(num_units, num_types)

    def train(self, sess, x_seq, t_seq, N, T, num_epochs):
        batch_size = tf.shape(x_seq)[0]

        inter_t_seq = tf.map_fn(
            fn=lambda idx: tf.random.uniform([N], 0., T[idx]),
            elems=tf.range(batch_size),
            dtype=tf.float32
        )

        h_base, h_inter = self._trainer.train(x_seq, t_seq, inter_t_seq, T)
        likelihood = self._get_likelihood(x_seq, h_base, h_inter, N, T)

        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(-likelihood)

        sess.run(tf.initializers.global_variables())

        progress_bar = trange(num_epochs)
        for epoch_id in progress_bar:
            _, train_lhd = sess.run([train_step, likelihood])
            progress_bar.set_postfix_str(f'Likelihood: {train_lhd}')

    def _get_likelihood(self, x_seq, h_base, h_inter, N, T):
        _, pos_lhd = tf.map_fn(
            fn=lambda xh: (0, self._intensity.get_intensity_for_type(*xh)),
            elems=(
                tf.transpose(x_seq),
                tf.transpose(h_base, [1, 0, 2])
            )
        )
        pos_lhd = tf.reduce_sum(tf.log(pos_lhd), axis=0)

        neg_lhd = self._intensity.get_all_intesities(h_inter)
        neg_lhd = T * tf.einsum('ijk->i', neg_lhd) / N

        return tf.reduce_mean(pos_lhd - neg_lhd)
