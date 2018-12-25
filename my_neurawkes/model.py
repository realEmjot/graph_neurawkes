import tensorflow as tf
from tqdm import trange

from .cont_lstm import ContLSTMCell, ContLSTMTrainer
from .intensity import Intensity


class Neurawkes:
    def __init__(self, num_units, num_types):
        self._cell = ContLSTMCell(num_units, num_types + 1)
        self._trainer = ContLSTMTrainer(self._cell)
        self._intensity = Intensity(num_units, num_types)

    def train(self, sess, dataset, N_ratio, num_epochs, batch_size):
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([None], [None], []),
            padding_values=(-1, 0., 0.)
        )
        iterator = dataset.make_initializable_iterator()
        x_seq, t_seq, T = iterator.get_next()

        T_max = tf.reduce_max(T)
        t_seq = tf.where(
            tf.not_equal(x_seq, -1),
            x=t_seq,
            y=tf.fill(tf.shape(t_seq), T_max)
        )
        N = N_ratio * tf.cast(tf.shape(x_seq)[1], tf.float32)

        inter_t_seq = tf.map_fn(
            fn=lambda T: tf.random.uniform([tf.cast(N, tf.int32)], 0., T),
            elems=T
        )

        h_base, h_inter = self._trainer.train(x_seq, t_seq, inter_t_seq, T_max)
        likelihood = self._get_likelihood(x_seq, h_base, h_inter, N, T)

        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(-likelihood)

        sess.run(tf.initializers.global_variables())

        progress_bar = trange(num_epochs)
        for epoch_id in progress_bar:
            sess.run(iterator.initializer)
            while True:
                try:
                    _, train_lhd = sess.run([train_step, likelihood])
                    progress_bar.set_postfix_str(f'Likelihood: {train_lhd}')
                except tf.errors.OutOfRangeError:
                    break

    def _get_likelihood(self, x_seq, h_base, h_inter, N, T):
        def get_pos_lhd_for_batch_elem(x, h):
            mask = tf.not_equal(x, -1)
            x = tf.boolean_mask(x, mask)
            h = tf.boolean_mask(h, mask)
            return 0, tf.reduce_sum(tf.log(
                self._intensity.get_intensity_for_type(x, h)))

        _, pos_lhd = tf.map_fn(
            fn=lambda xh: get_pos_lhd_for_batch_elem(*xh),
            elems=(
                x_seq,
                h_base
            )
        )

        neg_lhd = self._intensity.get_all_intesities(h_inter)
        neg_lhd = T * tf.einsum('ijk->i', neg_lhd) / N

        return tf.reduce_mean(pos_lhd - neg_lhd)
