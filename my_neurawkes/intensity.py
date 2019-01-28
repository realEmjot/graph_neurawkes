import tensorflow as tf
from .cont_lstm import ContLSTMCell, ContLSTMTrainer


#TODO use num_units and num_types from __init__ in all methods
class Intensity:
    """
    Class for obtaining event intensities from CSTM hidden states.
    """
    def __init__(self, num_units, num_types):
        """
        Parameters:
            num_units (int): size of modified CSTM model
            num_types (int): number of different event types in data
        """
        self.W, self.s = self._create_type_variables(num_units, num_types)

    def get_intensity_for_type(self, type_ids, cstm_hs):
        """
        Method for obtaining intensity for each event in given event stream.
        Parameters:
            type_ids (tf.Tensor): vector of event types
            cstm_hs (tf.Tensor): matrix of CSTM hidden states for every element
            in type_ids param; its first dim is the same as the size of
            type_ids param, and its second dim is equal to num_units
            parameter of __init__ method
        Returns:
            (tf.Tensor): vector of intensity values for every type in
                type_ids param
        """
        w = tf.gather(self.W, type_ids)
        s = tf.gather(self.s, type_ids)
        raw_intensity = tf.reduce_sum(w * cstm_hs, axis=1)
        return s * tf.nn.softplus(raw_intensity / s)

    def get_all_intesities(self, ctsm_Hs):
        """
        TODO
        """
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
