import tensorflow as tf
from . import utils


class Intensity(utils.VariablesContainer):
    def __init__(self, num_units, num_types):
        self.num_types = num_types  # used by generator
        self.W, self._raw_s = self._create_type_variables(num_units, num_types)

    @property
    def s(self):
        return tf.math.abs(self._raw_s)

    def get_variables_list(self):
        return [self.W, self._raw_s]

    def get_intensity_for_type(self, type_ids, cstm_hs):
        w = tf.gather(self.W, type_ids)
        s = tf.gather(self.s, type_ids)
        raw_intensity = tf.reduce_sum(w * cstm_hs, axis=1)
        return self.apply_transfer_function(raw_intensity, s=s)

    def get_all_intensities(self, ctsm_Hs):
        raw_intensities = tf.einsum('ijk,lk->ijl', ctsm_Hs, self.W)
        return self.apply_transfer_function(raw_intensities)

    def apply_transfer_function(self, input_, s=None):
        if s is None:
            s = self.s
        return s * tf.nn.softplus(input_ / s)

    def _create_type_variables(self, num_units, num_types):
        with tf.variable_scope('type_vars'):
            W = tf.get_variable('W', [num_types, num_units])
            raw_s = tf.get_variable(
                's',
                [num_types],
                initializer=tf.initializers.ones
            )
        return W, raw_s


class GraphIntensity(utils.VariablesContainer):
    def __init__(self, num_units, num_vertices, vstate_len, self_links=False):
        self.num_vertices = num_vertices
        self._self_links = self_links

        self._raw_s, self.W, self.W_self = self._create_type_variables(
            num_units, num_vertices, vstate_len, self_links)

    @property
    def s(self):
        return tf.math.abs(self._raw_s)

    def get_variables_list(self):
        var_list = [self._raw_s, self.W]
        if self._self_links:
            var_list.append(self.W_self)
        return var_list

    def get_intensity_for_pairs_sequence(self, x1_seq, x2_seq, cstm_hs):
        W1 = tf.gather(self.W, x1_seq)
        s = tf.gather(self.s, x1_seq)

        v1_states = tf.einsum('ijk,ik->ij', W1, cstm_hs)
        all_states = tf.einsum('ijk,lk->lij', self.W, cstm_hs)

        indices = tf.range(self.num_vertices)
        if self._self_links:
            def choose_state(normal_s, self_s, x1):
                return tf.where(
                    x=normal_s,
                    y=self_s,
                    condition=tf.not_equal(indices, x1)
                )

            all_self_states = tf.einsum('ijk,lk->lij', self.W_self, cstm_hs)
            all_states = tf.map_fn(
                fn=lambda ssx: choose_state(*ssx),
                elems=(all_states, all_self_states, x1_seq),
                dtype=all_states.dtype
            )
        else:
            def apply_mask(s, x):
                return tf.boolean_mask(s, tf.not_equal(indices, x))

            all_states = tf.map_fn(
                fn=lambda sx: apply_mask(*sx),
                elems=(all_states, x1_seq),
                dtype=all_states.dtype
            )

        dot_products = tf.einsum('ijk,ik->ij', all_states, v1_states)
        v1_norms = tf.sqrt(tf.reduce_sum(tf.pow(v1_states, 2), axis=-1))
        all_norms = tf.sqrt(tf.reduce_sum(tf.pow(all_states, 2), axis=-1))
        multiplied_norms = tf.expand_dims(v1_norms, -1) * all_norms

        cos_sims = dot_products / multiplied_norms
        softmax_sims = tf.math.softmax(cos_sims, axis=-1)

        v2_indices = x2_seq
        if not self._self_links:
            v2_indices -= tf.cast(x1_seq < x2_seq, tf.int32)

        v2_indices_mask = tf.one_hot(
            v2_indices, self.num_vertices - int(not self._self_links),
            on_value=True, off_value=False)
        v2_sims = tf.boolean_mask(softmax_sims, v2_indices_mask)

        return self.apply_transfer_function(v1_norms, s=s) * v2_sims

    def get_intensity_for_every_vertex_in_batchseq(self, ctsm_Hs):
        all_states = tf.einsum('ijk,lmk->lmij', self.W, ctsm_Hs)
        all_norms = tf.sqrt(tf.reduce_sum(tf.pow(all_states, 2), axis=-1))
        return self.apply_transfer_function(all_norms)

    def apply_transfer_function(self, input_, s=None):
        if s is None:
            s = self.s
        return s * tf.nn.softplus(input_ / s)

    def _create_type_variables(self, num_units, num_vertices, vstate_len, self_links=False):
        with tf.variable_scope('vertices_vars'):
            W = tf.get_variable('W', [num_vertices, vstate_len, num_units])
            raw_s = tf.get_variable(
                's',
                [num_vertices],
                initializer=tf.initializers.ones
            )

            if self_links:
                W_self = tf.get_variable('W_self', [num_vertices, vstate_len, num_units])
            else:
                W_self = None

        return raw_s, W, W_self
