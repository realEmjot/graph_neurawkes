import math

import tensorflow as tf
from tqdm import tqdm, trange

from .cont_lstm import ContLSTMCell, ContLSTMTrainer
from .intensity import Intensity


class Neurawkes:
    """
    This class is an implementation of Neural Hawkes model (proposed by _ in
    their paper _), written in Google Tensorflow.
    """
    def __init__(self, num_units, num_types):
        """
        Parameters:
            num_units (int): size of modified LSTM model (CSTM); higher number
                             means more expressive model, but training takes
                             longer
            num_types (int): number of different event types in data
        """
        self._cell = ContLSTMCell(num_units, num_types + 1)
        self._trainer = ContLSTMTrainer(self._cell)
        self._intensity = Intensity(num_units, num_types)

    def train(self, sess, dataset, N_ratio, num_epochs, batch_size,
              dataset_size=None):
        """
        Model training.
        Parameters:
            sess (tf.Session): Tensorflow Session object
            dataset (tf.Dataset): yet unbatched tf.Dataset, containing training
                data; each element should be a tuple of three elements:
                (event type sequence, event time-since-start sequence,
                end-of-sequence time)
            N_ratio (double): positive number, designated ratio between
                negative and positive examples for likelihood function
            num_epochs (int)
            batch_size (int)
            dataset_size (int) optional
        """
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
            fn=lambda T: tf.random.uniform([tf.cast(N, tf.int32)], 1e-10, T),  # TODO hack
            elems=T
        )

        h_base, h_inter = self._trainer.train(x_seq, t_seq, inter_t_seq, T_max)
        likelihood = self._get_likelihood(x_seq, h_base, h_inter, N, T)

        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(-likelihood)

        sess.run(tf.initializers.global_variables())

        epoch_progress_bar = trange(num_epochs)
        if dataset_size:
            num_iters = math.ceil(dataset_size / batch_size)

        for epoch_id in epoch_progress_bar:
            if dataset_size:
                iter_progress_bar = tqdm(total=num_iters, leave=False)

            sess.run(iterator.initializer)
            while True:
                try:
                    _, train_lhd = sess.run([train_step, likelihood])
                    epoch_progress_bar.set_postfix_str(f'Epoch: {epoch_id}\tLikelihood: {train_lhd}')
                    if dataset_size:
                        iter_progress_bar.update()
                except tf.errors.OutOfRangeError:
                    if dataset_size:
                        iter_progress_bar.close()
                    break

    def _get_likelihood(self, x_seq, h_base, h_inter, N, T):
        def get_pos_lhd_for_batch_elem(x, h):
            mask = tf.not_equal(x, -1)
            x = tf.boolean_mask(x, mask)
            h = tf.boolean_mask(h, mask)
            return tf.math.count_nonzero(mask, dtype=tf.float32), tf.reduce_sum(tf.log(
                self._intensity.get_intensity_for_type(x, h)))

        seq_lens, pos_lhd = tf.map_fn(
            fn=lambda xh: get_pos_lhd_for_batch_elem(*xh),
            elems=(
                x_seq,
                h_base
            ),
            dtype=(tf.float32, tf.float32)
        )

        neg_lhd = self._intensity.get_all_intesities(h_inter)
        neg_lhd = T * tf.einsum('ijk->i', neg_lhd) / N

        return tf.reduce_mean((pos_lhd - neg_lhd) / seq_lens)  # likelihood per event
