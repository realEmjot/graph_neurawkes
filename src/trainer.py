import tensorflow as tf


class ContLSTMTrainer:
    def __init__(self, cell):
        self.cell = cell

        self.num_units = cell.num_units
        self.elem_size = cell.elem_size

    def get_hidden_states_for_training(self, x_seq, t_seq, inter_t_seq, T_max):
        batch_size = tf.shape(x_seq)[0]

        bos_x = tf.one_hot(tf.zeros([batch_size], tf.int32), self.elem_size)
        bos_t = tf.zeros([batch_size])

        init_state = (
            tf.zeros([batch_size, self.num_units], name='init_c'),
            tf.zeros([batch_size, self.num_units], name='init_c_base'),
            tf.zeros([batch_size, self.num_units], name='init_h')
        )

        inter_t_mask = self._get_inter_t_mask(t_seq, inter_t_seq, T_max)
        
        #########################

        graph_vars = self.cell.build_graph(bos_x, bos_t, *init_state)
        init_inter_h, init_slices = self._get_inter_h_for_step(
            0, inter_t_seq, inter_t_mask, graph_vars
        )

        #########################

        h_acc = tf.zeros([batch_size, 0, self.num_units])
        inter_h_acc = init_inter_h
        slice_acc = tf.expand_dims(init_slices, 1)

        init_counter = tf.constant(0)
        _, _, h_acc, inter_h_acc, slice_acc = tf.while_loop(
            cond=lambda idx, *_: idx < tf.shape(x_seq)[1],
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
        curr_inter_t_mask = inter_t_mask[idx]
        t_init = graph_vars[0]

        inter_t_seq = tf.where(
            condition=curr_inter_t_mask,
            x=inter_t_seq,
            y=tf.tile(tf.expand_dims(t_init, -1), [1, tf.shape(inter_t_seq)[1]])
        )
        _, _, inter_h = self.cell.get_time_dependent_vars(
            tf.transpose(inter_t_seq),
            graph_vars
        )
        inter_h = tf.transpose(inter_h, [1, 0 ,2])

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
        inc = tf.scan(  # TODO check if this could be replaced by tf.cumsum
            fn=lambda acc, x: acc + x,
            elems=counts
        )

        return tf.stack([
            tf.concat([[0], inc[:-1]], axis=0),
            inc
        ], axis=1)

    def _get_inter_t_mask(self, t_seq, inter_t_seq, T_max):
        inter_t_seq = tf.expand_dims(inter_t_seq, -1)

        left_bound_seq = tf.expand_dims(
            tf.pad(t_seq, [[0, 0], [1, 0]], constant_values=0), 1
        )
        right_bound_seq = tf.expand_dims(
            tf.pad(t_seq, [[0, 0], [0, 1]], constant_values=T_max), 1
        )

        mask = tf.logical_and(
            left_bound_seq < inter_t_seq, inter_t_seq <= right_bound_seq
        )
        return tf.transpose(mask, (2, 0, 1))
