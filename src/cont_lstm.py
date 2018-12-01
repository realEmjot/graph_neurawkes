import tensorflow as tf
import numpy as np

# no batches
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

    def build_graph(self, x, t_init, c_init, c_base_init, h_init): # t_init here for simplified unrolling
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

        return c_t, c_base, h_t
        
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

    def _c_func(self, t_init, t, c, c_base, decay):
        return c_base + (c - c_base) * tf.exp(-decay * (t - t_init))

    def _h_func(self, c_t, o):
        return o * (2 * tf.sigmoid(2 * c_t) - 1)


# no batches
class ContLSTMTrainer:
    def __init__(self, cell):
        self.cell = cell

        self.num_units = cell.num_units
        self.elem_size = cell.elem_size

    def train(self, x_seq, t_seq, inter_t_seq):
        init_counter = tf.constant(0)

        # paper warning - check boundary conditions
        init_state = (
            tf.one_hot(0, self.elem_size), # BOS x
            tf.constant(0., dtype=tf.float32), # BOS t
            tf.zeros([self.cell.num_units], name='init_c'),
            tf.zeros([self.cell.num_units], name='init_c_base'),
            tf.zeros([self.cell.num_units], name='init_h')
        )
        init_graph_vars = self.cell.build_graph(*init_state)

        h_acc = tf.reshape(init_state[-1], [1, -1])
        inter_t_acc = h_acc

        _, _, _, h_acc, inter_t_acc = tf.while_loop(
            cond=lambda base_idx, inter_t_idx, *_: tf.logical_or(
                base_idx < x_seq.shape[0],
                inter_t_idx < inter_t_seq.shape[0]
            ),
            body=lambda base_idx, inter_t_idx, graph_vars, h_acc, inter_t_acc:
                self._train_loop_body(
                    base_idx,
                    inter_t_idx,
                    x_seq,
                    t_seq,
                    inter_t_seq,
                    init_graph_vars,
                    h_acc,
                    inter_t_acc
                ),
            loop_vars=[
                init_counter, init_counter, init_graph_vars, h_acc, inter_t_acc
            ],
            shape_invariants=[
                init_counter.shape,
                init_counter.shape,
                tuple(var.shape for var in init_graph_vars),
                tf.TensorShape([None, self.cell.num_units]),
                tf.TensorShape([None, self.cell.num_units])
            ]
        )

        return h_acc[1:], inter_t_acc[1:]

    def _train_loop_body(self, base_idx, inter_t_idx, x_seq, t_seq, inter_t_seq,
              graph_vars, h_acc, inter_t_acc):
        base_not_overflown = base_idx < x_seq.shape[0]
        inter_not_overflown = inter_t_idx < inter_t_seq.shape[0]
        base_or_inter = tf.cond(
            pred=tf.logical_and(base_not_overflown, inter_not_overflown),
            true_fn=lambda: t_seq[base_idx] < inter_t_seq[inter_t_idx],
            false_fn=lambda: base_not_overflown
        )

        return tf.cond(
            pred=base_or_inter,
            true_fn=lambda: self._base_case(base_idx, inter_t_idx, x_seq, t_seq,
                graph_vars, h_acc, inter_t_acc),
            false_fn=lambda: self._inter_case(base_idx, inter_t_idx, inter_t_seq,
                graph_vars, h_acc, inter_t_acc)
        )

    def _base_case(self, base_idx, inter_t_idx, x_seq, t_seq,
                  graph_vars, h_acc, inter_t_acc):
        x = x_seq[base_idx]
        t = t_seq[base_idx]

        c, c_base, h = self.cell.get_time_dependent_vars(t, graph_vars)
        graph_vars = self.cell.build_graph(x, t, c, c_base, h)
        h_acc = tf.concat([h_acc, [h]], axis=0)

        return (
            base_idx + 1,
            inter_t_idx,
            graph_vars,
            h_acc,
            inter_t_acc
        )

    def _inter_case(self, base_idx, inter_t_idx, inter_t_seq,
                   graph_vars, h_acc, inter_t_acc):
        t = inter_t_seq[inter_t_idx]

        _, _, h = self.cell.get_time_dependent_vars(t, graph_vars)
        inter_t_acc = tf.concat([inter_t_acc, [h]], axis=0)

        return (
            base_idx,
            inter_t_idx + 1,
            graph_vars,
            h_acc,
            inter_t_acc
        )
