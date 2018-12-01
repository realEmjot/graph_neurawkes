import tensorflow as tf
from cont_lstm import ContLSTMCell, ContLSTMTrainer


class Intensity:
    def __init__(self, num_units, num_types):
        types = range(1, num_types) # TODO think of a better place/way to ignore BOS type

        self.type_vars_list = [
            self._create_type_variables(num_units, str(idx))
                for idx in types
        ]

    def __call__(self, type_id, ctsm_h):
        w, s = self.type_vars_list[type_id]
        raw_intensity = tf.reduce_sum(w * ctsm_h)
        return s * tf.nn.softplus(raw_intensity / s)

    def _create_type_variables(self, num_units, name):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [num_units])
            s = tf.math.abs(
                tf.get_variable('s', [], initializer=tf.initializers.ones)
            )
        return w, s
