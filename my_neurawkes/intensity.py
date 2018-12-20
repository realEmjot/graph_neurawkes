import tensorflow as tf
from .cont_lstm import ContLSTMCell, ContLSTMTrainer


class Intensity:
    def __init__(self, num_units, num_types):
        self.W, self.s = self._create_type_variables(num_units, num_types)

    def get_intensity_for_type(self, type_ids, ctsm_hs):
        w = tf.gather(self.W, type_ids)
        s = tf.gather(self.s, type_ids)
        raw_intensity = tf.reduce_sum(w * ctsm_hs, axis=1)
        return s * tf.nn.softplus(raw_intensity / s)

    def get_all_intesities(self, ctsm_Hs):
        raw_intensities = tf.einsum('ijk,lk->ijl', ctsm_Hs, self.W)
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
