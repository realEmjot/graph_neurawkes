import tensorflow as tf
import numpy as np


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

    def _c_func(self, t_init, t, c, c_base, decay):
        return c_base + (c - c_base) * tf.exp(
            -decay * tf.expand_dims(t - t_init, -1)
        )

    def _h_func(self, c_t, o):
        return o * (2 * tf.sigmoid(2 * c_t) - 1)



class ContLSTMTrainer:
    def __init__(self, cell):
        self.cell = cell

        self.num_units = cell.num_units
        self.elem_size = cell.elem_size

    def train(self, x_seq, t_seq, inter_t_seq, T):
        batch_size = tf.shape(x_seq)[0]
        x_seq = tf.one_hot(x_seq + 1, self.elem_size)

        bos_x = tf.zeros([batch_size, self.elem_size])
        bos_t = tf.zeros([batch_size])

        init_state = (
            tf.zeros([batch_size, self.num_units], name='init_c'),
            tf.zeros([batch_size, self.num_units], name='init_c_base'),
            tf.zeros([batch_size, self.num_units], name='init_h')
        )

        inter_t_mask = self._get_inter_t_mask(t_seq, inter_t_seq, T)
        
        #########################

        graph_vars = self.cell.build_graph(bos_x, bos_t, *init_state)
        init_inter_h, init_count = self._get_inter_h_for_step(
            0, inter_t_seq, inter_t_mask, graph_vars
        )

        #########################

        h_acc = tf.zeros([batch_size, 0, self.num_units])
        inter_h_acc = init_inter_h
        slice_acc = tf.expand_dims(init_count, 1)

        init_counter = tf.constant(0)
        _, _, h_acc, inter_h_acc, slice_acc = tf.while_loop(
            cond=lambda idx, *_: idx < x_seq.shape[1],
            body=lambda idx, graph_vars, h_acc, inter_h_acc, slice_acc:
                self._train_loop(
                    idx,
                    x_seq,
                    t_seq,
                    inter_t_seq,
                    inter_t_mask[1:],
                    graph_vars,
                    h_acc,
                    inter_h_acc,
                    slice_acc
                ),
            loop_vars=[
                init_counter, graph_vars, h_acc, inter_h_acc, slice_acc
            ],
            shape_invariants=[
                init_counter.shape,
                tuple(var.shape for var in graph_vars),
                tf.TensorShape([h_acc.shape[0], None, h_acc.shape[2]]),
                tf.TensorShape([None, inter_h_acc.shape[1]]),
                tf.TensorShape([slice_acc.shape[0], None, slice_acc.shape[2]])
            ]
        )

        return h_acc, self._reshape_inter_h_acc(inter_h_acc, slice_acc)


    def _train_loop(self, idx, x_seq, t_seq, inter_t_seq, inter_t_mask,
                    graph_vars, h_acc, inter_h_acc, slice_acc):
        x = x_seq[:, idx]
        t = t_seq[:, idx]

        c, c_base, h = self.cell.get_time_dependent_vars(t, graph_vars)
        h_acc = tf.concat([h_acc, tf.expand_dims(h, 1)], axis=1)
        graph_vars = self.cell.build_graph(x, t, c, c_base, h)

        inter_h, slices = self._get_inter_h_for_step(
            idx, inter_t_seq, inter_t_mask, graph_vars
        )
        inter_h_acc = tf.concat([inter_h_acc, inter_h], axis=0)
        slices += slice_acc[-1, -1, -1]
        slice_acc = tf.concat([slice_acc, tf.expand_dims(slices, 1)], axis=1)

        return idx + 1, graph_vars, h_acc, inter_h_acc, slice_acc

    def _get_inter_h_for_step(self, idx, inter_t_seq, inter_t_mask, graph_vars):
        _, _, inter_h = self.cell.get_time_dependent_vars(
            tf.transpose(inter_t_seq),
            graph_vars
        )
        inter_h = tf.transpose(inter_h, [1, 0 ,2])

        curr_inter_t_mask = inter_t_mask[idx]

        curr_counts = tf.count_nonzero(curr_inter_t_mask, axis=1)
        curr_slices = self._convert_counts_to_slices(curr_counts)

        return tf.boolean_mask(inter_h, curr_inter_t_mask), curr_slices

    def _reshape_inter_h_acc(self, inter_h_acc, slice_acc): #TODO probably slow
        def gather(idx, acc, slices):
            s = slices[idx]
            return idx + 1, tf.concat([acc, inter_h_acc[s[0]:s[1]]], axis=0)

        init_idx = tf.constant(0)
        acc = tf.zeros([0, self.num_units])
        slices_len = tf.shape(slice_acc)[1]

        return tf.map_fn(
            fn=lambda slices: tf.while_loop(
                cond=lambda idx, *_: idx < slices_len,
                body=lambda idx, acc: gather(idx, acc, slices),
                loop_vars=[init_idx, acc],
                shape_invariants=[
                    init_idx.shape, tf.TensorShape([None, acc.shape[1]])
                ]
            )[1],
            elems=slice_acc,
            dtype=inter_h_acc.dtype
        ) 

    def _convert_counts_to_slices(self, counts):
        inc = tf.scan(
            fn=lambda acc, x: acc + x,
            elems=counts,
            initializer=tf.constant(0, dtype=tf.int64)
        )

        return tf.stack([
            tf.concat([[0], inc[:-1]], axis=0),
            inc
        ], axis=1)

    def _get_inter_t_mask(self, t_seq, inter_t_seq, T):
        transformed_t_seq = tf.expand_dims(
            tf.transpose(
                tf.pad(t_seq, [[0, 0], [1, 0]])
            ),
            -1
        )

        def get_single_mask(curr_t, next_t):
            return tf.logical_and(curr_t < inter_t_seq, inter_t_seq <= next_t)

        return tf.map_fn(
            fn=lambda elems: get_single_mask(*elems),
            elems=(
                transformed_t_seq,
                tf.concat(
                    [transformed_t_seq[1:], [tf.expand_dims(T, -1)]],
                    axis=0
                )
            ),
            dtype=tf.bool
        )
    