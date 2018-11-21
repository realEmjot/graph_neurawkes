import tensorflow as tf
import numpy as np

# no batches
class ContLSTMCell:
    def __init__(self, num_types, num_units):
        self.num_types = num_types # TODO possibly unneeded
        self.num_units = num_units # TODO possibly unneeded

        self.i_vars = self._create_gate_variables(num_units, 'i_vars')
        self.f_vars = self._create_gate_variables(num_units, 'f_vars')
        self.z_vars = self._create_gate_variables(num_units, 'z_vars')
        self.o_vars = self._create_gate_variables(num_units, 'o_vars')

        self.i_base_vars = self._create_gate_variables(num_units, 'i_base_vars')
        self.f_base_vars = self._create_gate_variables(num_units, 'f_base_vars')

        self.decay_vars = self._create_gate_variables(num_units, 'decay_vars')
        
    def _create_gate_variables(self, num_units, name): # TODO initializers
        with tf.variable_scope(name):
            W = tf.get_variable('W', [None, num_units]) # TODO possibly bad shape
            U = tf.get_variable('U', [num_units] * 2)
            d = tf.get_variable('d')
        return W, U, d

    def _create_gate(self, W, U, d, x, h_init, activation, name):
        return activation(W * x + U * h_init + d, name=name)

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

        return c_t, h_t


# no batches
class ContLSTMNetwork:
    def __init__(self, cell):
        self.cell = cell

    # def __call__(self, xs, ts):
    #     states = [(np.zeros(self.cell.num_units),) * 2] # warning

    #     for x, t in zip(xs, ts): # TODO xs and ts should be tuples for now
    #         state = self.cell(x, t, states[-1])
    #         states.append(state)

    #     return states

    def __call__(self, x_seq, t_seq, training=True): # TODO training
        states = [(np.zeros(self.cell.num_units),) * 2] # warning

        init_counter = tf.constant(0)
        tf.while_loop(
            cond=lambda idx: idx < x_seq.shape[0],
            body=lambda idx: idx + 1, # TODO
            loop_vars=[init_counter],
            parallel_iterations=1
        )

    def _network_loop_body(self, idx, x_seq, t_seq, states):
        x = x_seq[idx]
        t = t_seq[idx]
        t_init = None # TODO
        c_init, c_base_init, h_init = states[-1]

        state = self.cell(x, t, t_init, c_init, c_base_init, h_init)
        states.append(state)

        return idx + 1
