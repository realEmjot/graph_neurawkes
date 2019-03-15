import tensorflow as tf


class ContLSTMCell:
    def __init__(self, num_units, elem_size):
        self.num_units = num_units
        self.elem_size = elem_size

        self.i_vars = self._create_gate_variables(num_units, elem_size, 'i_vars')
        self.f_vars = self._create_gate_variables(num_units, elem_size, 'f_vars')
        self.z_vars = self._create_gate_variables(num_units, elem_size, 'z_vars')
        self.o_vars = self._create_gate_variables(num_units, elem_size, 'o_vars')

        self.i_base_vars = self._create_gate_variables(num_units, elem_size, 'i_base_vars')
        self.f_base_vars = self._create_gate_variables(num_units, elem_size, 'f_base_vars')

        self.decay_vars = self._create_gate_variables(num_units, elem_size, 'decay_vars')

    def build_graph(self, x, t_init, c_init, c_base_init, h_init): # t_init here for convenience
        i = self._create_gate(*self.i_vars, x, h_init, tf.sigmoid, 'i_gate')
        f = self._create_gate(*self.f_vars, x, h_init, tf.sigmoid, 'f_gate')
        z = self._create_gate(*self.z_vars, x, h_init, tf.tanh, 'z_gate')
        o = self._create_gate(*self.o_vars, x, h_init, tf.sigmoid, 'o_gate')

        i_base = self._create_gate(*self.i_base_vars, x, h_init, tf.sigmoid, 'i_base_gate')
        f_base = self._create_gate(*self.f_base_vars, x, h_init, tf.sigmoid, 'f_base_gate')

        c = f * c_init + i * z
        c_base = f_base * c_base_init + i_base * z
        decay = self._create_gate(*self.decay_vars, x, h_init, tf.nn.softplus, 'decay') # warning: softplus?

        return t_init, c, c_base, decay, o

    def get_time_dependent_vars(self, t, graph_vars):
        t_init, c, c_base, decay, o = graph_vars

        c_t = self._c_func(t, t_init, c, c_base, decay)
        h_t = self._h_func(c_t, o)

        return c_t, c_base, h_t # c_base returned here for convenience
        
    def _create_gate_variables(self, num_units, elem_size, name): # TODO parametrize initializers
        with tf.variable_scope(name, initializer=tf.glorot_normal_initializer()):
            W = tf.get_variable('W', [elem_size, num_units])
            U = tf.get_variable('U', [num_units] * 2)
            d = tf.get_variable('d', [num_units])
        return W, U, d

    def _create_gate(self, W, U, d, x, h_init, activation, name):
        return activation(
            tf.matmul(x, W) + tf.matmul(h_init, U) + d,
            name=name
        )

    def _c_func(self, t, t_init, c, c_base, decay):
        return c_base + (c - c_base) * tf.exp(
            -decay * tf.expand_dims(t - t_init, -1)
        )

    def _h_func(self, c_t, o):
        return o * (2 * tf.sigmoid(2 * c_t) - 1)
