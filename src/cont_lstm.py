import tensorflow as tf
import numpy as np

# no batches
class ContLSTMCell:
    def __init__(self, num_units, elem_size):
        self.num_units = num_units # TODO possibly unneeded

        self.i_vars = self._create_gate_variables(num_units, elem_size, 'i_vars')
        self.f_vars = self._create_gate_variables(num_units, elem_size, 'f_vars')
        self.z_vars = self._create_gate_variables(num_units, elem_size, 'z_vars')
        self.o_vars = self._create_gate_variables(num_units, elem_size, 'o_vars')

        self.i_base_vars = self._create_gate_variables(num_units, elem_size, 'i_base_vars')
        self.f_base_vars = self._create_gate_variables(num_units, elem_size, 'f_base_vars')

        self.decay_vars = self._create_gate_variables(num_units, elem_size, 'decay_vars')
        
    def _create_gate_variables(self, num_units, elem_size, name): # TODO parametrize initializers
        with tf.variable_scope(name, initializer=tf.glorot_normal_initializer()):
            W = tf.get_variable('W', [num_units, elem_size])
            U = tf.get_variable('U', [num_units] * 2)
            d = tf.get_variable('d', [num_units])
        return W, U, d

    def _create_gate(self, W, U, d, x, h_init, activation, name):
        return activation(
            tf.squeeze( # TODO check if this is efficient
                tf.matmul(W, tf.reshape(x, [-1, 1])) + tf.matmul(U, tf.reshape(h_init, [-1, 1]))
            ) + d,
            name=name
        )

    def _build_graph(self, x, c_init, c_base_init, h_init):
        i = self._create_gate(*self.i_vars, x, h_init, tf.sigmoid, 'i_gate')
        f = self._create_gate(*self.f_vars, x, h_init, tf.sigmoid, 'f_gate')
        z = self._create_gate(*self.z_vars, x, h_init, tf.tanh, 'z_gate')
        o = self._create_gate(*self.o_vars, x, h_init, tf.sigmoid, 'o_gate')

        i_base = self._create_gate(*self.i_base_vars, x, h_init, tf.sigmoid, 'i_base_gate')
        f_base = self._create_gate(*self.f_base_vars, x, h_init, tf.sigmoid, 'f_base_gate')

        c = f * c_init + i * z
        c_base = f_base * c_base_init + i_base * z
        decay = self._create_gate(*self.decay_vars, x, h_init, tf.nn.softplus, 'decay') # warning: softplus?

        return c, c_base, decay, o

    def _c_func(self, t_init, t, c, c_base, decay):
        return c_base + (c - c_base) * tf.exp(-decay * (t - t_init))

    def _h_func(self, c_t, o):
        return o * (2 * tf.sigmoid(2 * c_t) - 1)

    def __call__(self, x, t_init, t, c_init, c_base_init, h_init):
        c, c_base, decay, o = self._build_graph(x, c_init, c_base_init, h_init)

        c_t = self._c_func(t, t_init, c, c_base, decay)
        h_t = self._h_func(c_t, o)

        return c_t, c_base, h_t


# no batches
class ContLSTMNetwork:
    def __init__(self, cell):
        self.cell = cell

    def __call__(self, x_seq, t_seq, training=True): # TODO training
        init_counter = tf.constant(0)

        # paper warning - check boundary conditions
        init_state = (
            tf.zeros([self.cell.num_units], name='init_c'),
            tf.zeros([self.cell.num_units], name='init_c_base'),
            tf.zeros([self.cell.num_units], name='init_h')
        )

        h_acc = tf.reshape(init_state[2], [1, -1])

        _, _, h_acc = tf.while_loop(
            cond=lambda idx, _, __: idx < x_seq.shape[0],
            body=lambda idx, prev_state, h_acc:
                self._network_loop_body(idx, x_seq, t_seq, prev_state, h_acc),
            loop_vars=[init_counter, init_state, h_acc],
            shape_invariants=[
                init_counter.shape,
                (init_state[0].shape,) * 3,
                tf.TensorShape([None, self.cell.num_units])
            ]
        )

        return h_acc

    def _network_loop_body(self, idx, x_seq, t_seq, prev_state, h_acc):
        x = x_seq[idx]
        t_init = t_seq[idx]
        t = t_seq[idx + 1]

        c_init, c_base_init, h_init = prev_state

        c, c_base, h = self.cell(x, t, t_init, c_init, c_base_init, h_init)
        h_acc = tf.concat([h_acc, [h]], axis=0)

        return idx + 1, (c, c_base, h), h_acc
