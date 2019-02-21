import tensorflow as tf
from .cont_lstm import ContLSTMCell, ContLSTMTrainer



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


class GraphIntensity:
    def __init__(self, num_units, num_vertices, vstate_len, self_links=False):
        self._num_vertices = num_vertices
        self._self_links = self_links

        self.W, self.s = self._create_type_variables(
            num_units, num_vertices, vstate_len)

    def get_intensity_for_specific_pairs(self, x1_seq, x2_seq, cstm_hs):
        W1 = tf.gather(self.W, x1_seq)
        s = tf.gather(self.s, x1_seq)

        v1_states = tf.einsum('ijk,ik->ij', W1, cstm_hs)
        all_states = tf.einsum('ijk,lk->lij', self.W, cstm_hs)

        if self._self_links:
            raise NotImplementedError
        else:
            indices = tf.range(self._num_vertices)
            def lol(s, x):
                return tf.boolean_mask(s, tf.not_equal(indices, x))

            all_states = tf.map_fn(
                fn=lambda sx: lol(*sx),
                elems=(all_states, x1_seq),
                dtype=all_states.dtype
            )

        dot_products = tf.einsum('ijk,ik->ij', all_states, v1_states)
        v1_norms = tf.sqrt(tf.reduce_sum(tf.pow(v1_states, 2), axis=-1))
        all_norms = tf.sqrt(tf.reduce_sum(tf.pow(all_states, 2), axis=-1))
        norms = tf.expand_dims(v1_norms, -1) * all_norms

        cos_sims = dot_products / norms
        softmax_sims = tf.math.softmax(cos_sims, axis=-1)

        v2_indices = x2_seq
        if not self._self_links:
            v2_indices -= tf.cast(x1_seq < x2_seq, tf.int32)

        v2_indices_mask = tf.one_hot(
            v2_indices, self._num_vertices - 1, on_value=True, off_value=False)
        v2_sims = tf.boolean_mask(softmax_sims, v2_indices_mask)

        return s * tf.nn.softplus(v1_norms / s) * v2_sims

    def get_all_intesities(self, ctsm_Hs):
        all_states = tf.einsum('ijk,lmk->lmij', self.W, ctsm_Hs)
        all_norms = tf.sqrt(tf.reduce_sum(tf.pow(all_states, 2), axis=-1))
        return self.s * tf.nn.softplus(all_norms / self.s)

    def _create_type_variables(self, num_units, num_vertices, vstate_len):
        with tf.variable_scope('vertices_vars'):
            W = tf.get_variable('W', [num_vertices, vstate_len, num_units])
            s = tf.math.abs(tf.get_variable(
                's',
                [num_vertices],
                initializer=tf.initializers.ones
            ))
        return W, s
