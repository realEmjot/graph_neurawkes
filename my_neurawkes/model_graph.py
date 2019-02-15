import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange

from .cont_lstm import ContLSTMCell, ContLSTMTrainer
from .intensity_graph import GraphIntensity


class GraphNeurawkes:

    def __init__(self, num_units, num_vertices, vstate_len):
        self._num_vertices = num_vertices

        self._cell = ContLSTMCell(num_units, 2 * num_vertices + 1)
        self._trainer = ContLSTMTrainer(self._cell)
        self._intensity = GraphIntensity(num_units, num_vertices, vstate_len)

    def train(self, sess, dataset, N_ratio, num_epochs, batch_size,
              dataset_size=None, val_ratio=None):

        iterator = tf.data.Iterator.from_structure(
            dataset.output_types,
            ([None, None], [None, None], [None, None], [None])
        )
        x1_seq, x2_seq, t_seq, T = iterator.get_next()

        T_max = tf.reduce_max(T)
        t_seq = tf.where(
            tf.not_equal(t_seq, -1.),
            x=t_seq,
            y=tf.fill(tf.shape(t_seq), T_max)
        )
        N = N_ratio * tf.cast(tf.shape(t_seq)[1], tf.float32)

        inter_t_seq = tf.map_fn(
            fn=lambda T: tf.random.uniform([tf.cast(N, tf.int32)], 1e-10, T),  # TODO hack
            elems=T
        )

        x_seq_oh = tf.concat(
            [
                tf.one_hot(x1_seq + 1, self._num_vertices + 1),
                tf.one_hot(x2_seq, self._num_vertices)
            ],
            axis=-1
        )
        h_base, h_inter = self._trainer.get_hidden_states_for_training(
            x_seq_oh, t_seq, inter_t_seq, T_max)
        likelihood = self._get_likelihood(x1_seq, x2_seq, h_base, h_inter, N, T)

        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(-likelihood)

        sess.run(tf.initializers.global_variables())

        epoch_progress_bar = range(num_epochs)
        if dataset_size:
            num_iters = math.ceil(dataset_size / batch_size)
        else:
            epoch_progress_bar = trange(num_epochs, desc='Epoch')

        val_init_op = None
        if dataset_size and val_ratio:
            val_size = int(dataset_size * val_ratio)
            num_iters = math.ceil((dataset_size - val_size) / batch_size)
            num_iters_val = math.ceil((val_size) / batch_size)

            val_init_op = iterator.make_initializer(
                self._pad_dataset(dataset.take(val_size), batch_size)
            )
            dataset = dataset.skip(val_size)
        train_init_op = iterator.make_initializer(
            self._pad_dataset(dataset, batch_size)
        )

        train_lhd_acc = []
        val_lhd_acc = []
        for epoch_id in epoch_progress_bar:
            if dataset_size:
                iter_progress_bar = tqdm(total=num_iters, desc=f'Epoch {epoch_id}', leave=False)

            sess.run(train_init_op)
            while True:
                try:
                    _, train_lhd = sess.run([train_step, likelihood])
                except tf.errors.OutOfRangeError:
                    if dataset_size:
                        iter_progress_bar.close()
                    break

                train_lhd_acc.append(train_lhd)
                if dataset_size:
                    iter_progress_bar.update()
                    iter_progress_bar.set_postfix({'Likelihood': train_lhd})
                else:
                    epoch_progress_bar.set_postfix({'Likelihood': train_lhd})

            if val_init_op:
                sess.run(val_init_op)
                val_lhds = []
                val_progress_bar = tqdm(total=num_iters_val, desc=f'Validation...')
                while True:
                    try:
                        val_lhd = sess.run(likelihood)
                    except tf.errors.OutOfRangeError:
                        mean_val_lhd = np.array(val_lhds).mean()
                        val_lhd_acc.append(mean_val_lhd)
                        val_progress_bar.set_description(f'Epoch {epoch_id}')
                        val_progress_bar.set_postfix(
                            {'Validation likelihood': mean_val_lhd})
                        val_progress_bar.close()
                        break

                    val_progress_bar.update()
                    val_lhds.append(val_lhd)

        return train_lhd_acc, val_lhd_acc

    def _pad_dataset(self, ds, batch_size):
        return ds.padded_batch(
            batch_size,
            padded_shapes=([None], [None], [None], []),
            padding_values=(-1, -1, -1., 0.)
        )

    def _get_likelihood(self, x1_seq, x2_seq, h_base, h_inter, N, T):
        def get_pos_lhd_for_batch_elem(x1, x2, h):
            mask = tf.not_equal(x1, -1)
            x1 = tf.boolean_mask(x1, mask)
            x2 = tf.boolean_mask(x2, mask)
            h = tf.boolean_mask(h, mask)
            return tf.math.count_nonzero(mask, dtype=tf.float32), \
                   tf.reduce_sum(tf.log(
                        self._intensity.get_intensity_for_specific_pairs(x1, x2, h)))

        seq_lens, pos_lhd = tf.map_fn(
            fn=lambda xh: get_pos_lhd_for_batch_elem(*xh),
            elems=(
                x1_seq,
                x2_seq,
                h_base
            ),
            dtype=(tf.float32, tf.float32)
        )

        neg_lhd = self._intensity.get_all_intesities(h_inter)
        neg_lhd = T * tf.einsum('ijk->i', neg_lhd) / N

        return tf.reduce_mean((pos_lhd - neg_lhd) / seq_lens)  # likelihood per event
