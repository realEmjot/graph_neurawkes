import tensorflow as tf
from cont_lstm import ContLSTMCell, ContLSTMTrainer


class Intensity:
    def __init__(self, num_units, num_types):
        self.W, self.s = self._create_type_variables(num_units, num_types)

    def get_intensity_for_type(self, type_id, ctsm_h):
        w, s = self.W[type_id], self.s[type_id]
        raw_intensity = tf.reduce_sum(w * ctsm_h)
        return s * tf.nn.softplus(raw_intensity / s)

    def get_all_intesities(self, ctsm_h):
        raw_intensities = tf.squeeze(
            tf.matmul(self.W, tf.reshape(ctsm_h, [-1, 1]))
        )
        return self.s * tf.nn.softplus(raw_intensities / self.s)

    def get_all_intesities_batched(self, ctsm_H):
        raw_intensities = tf.matmul(ctsm_H, tf.transpose(self.W))
        return self.s * tf.nn.softplus(raw_intensities / self.s)

    def _create_type_variables(self, num_units, num_types):
        with tf.variable_scope('type_vars'):
            W = tf.get_variable('W', [num_types, num_units])
            s = tf.math.abs(tf.get_variable(
                's',
                [num_types],
                initializer=tf.initializers.ones
            ))
        return W, s
