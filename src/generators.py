import abc
import tensorflow as tf


class AbstractGenerator:
    def __init__(self, cell, intensity_obj):
        self.cell = cell
        self.intensity_obj = intensity_obj

    def generate_events(self, seed=None, max_events=None, max_time=None):  # TODO support for seed and self-links
        assert tf.executing_eagerly()
        assert (max_events, max_time) != (None, None)
        
        if seed is not None:
            pass #TODO

        graph_vars = self.cell.build_graph(
            tf.expand_dims(tf.one_hot(0, self.cell.elem_size), 0),  # bos x
            tf.expand_dims(0., 0),  # bos t
            tf.zeros([1, self.cell.num_units], name='init_c'),
            tf.zeros([1, self.cell.num_units], name='init_c_base'),
            tf.zeros([1, self.cell.num_units], name='init_h')
        )

        generated_sequence = []

        current_time = 0.
        c_t = graph_vars[1]

        while (max_events is None or len(generated_sequence) < max_events) and \
              (max_time is None or current_time < max_time):
            _, _, c_base, _, o = graph_vars
            upper_intensity_bound = self._get_intensity_upper_bound(c_t, c_base, o)

            current_time += tf.distributions.Exponential(upper_intensity_bound).sample()

            c_t, _, h_t = self.cell.get_time_dependent_vars(
                tf.expand_dims(current_time, 0),
                graph_vars
            )

            event_type = self._sample_event_type(h_t, upper_intensity_bound)
            if event_type is False:
                continue

            generated_sequence.append((*event_type, current_time.numpy()))

            x_oh = self._parse_type_to_one_hot(event_type)
            x_oh = tf.expand_dims(x_oh, 0)

            graph_vars = self.cell.build_graph(
                x_oh,
                tf.expand_dims(current_time, 0),
                c_t, c_base, h_t
            )
            c_t = graph_vars[1]

        return generated_sequence

    @abc.abstractmethod
    def _sample_event_type(self, h_t, upper_intensity_bound):
        pass

    @abc.abstractmethod
    def _get_intensity_upper_bound(self, c_t, c_base, o):
        pass

    @abc.abstractmethod
    def _parse_type_to_one_hot(self, event_type):
        pass


class NeurawkesGenerator(AbstractGenerator):
    def _sample_event_type(self, h_t, upper_intensity_bound):
        current_intensities = tf.einsum('ij,j->i', self.intensity_obj.W, tf.squeeze(h_t))
        current_intensities = self.intensity_obj.apply_transfer_function(current_intensities)

        assert upper_intensity_bound >= tf.reduce_sum(current_intensities) #TODO remove after a while

        random_val = tf.random.uniform([])
        if random_val * upper_intensity_bound > tf.reduce_sum(current_intensities):
            return False

        event_type = None  #TODO remove after a while
        for type_id, type_int_boundary in enumerate(tf.cumsum(current_intensities)):
            if random_val * upper_intensity_bound <= type_int_boundary:
                event_type = type_id
                break
        assert event_type is not None  #TODO remove after a while

        return [event_type]

    def _get_intensity_upper_bound(self, c, c_base, o):
        h1 = tf.squeeze(self.cell._h_func(c, o))
        h2 = tf.squeeze(self.cell._h_func(c_base, o))

        W = self.intensity_obj.W

        res1 = tf.einsum('ij,j->ij', W, h1)
        res2 = tf.einsum('ij,j->ij', W, h2)

        res_opt = tf.maximum(res1, res2)
        intensities_opt = self.intensity_obj.apply_transfer_function(
            tf.reduce_sum(res_opt, axis=1)
        )

        return tf.reduce_sum(intensities_opt)

    def _parse_type_to_one_hot(self, event_type):
        return tf.one_hot(event_type[0] + 1, self.intensity_obj.num_types + 1)


class GraphNeurawkesGenerator(AbstractGenerator):
    def _sample_event_type(self, h_t, upper_intensity_bound):
        current_vstates = tf.einsum('ijk,k->ij', self.intensity_obj.W, tf.squeeze(h_t))
        current_norms = tf.sqrt(tf.reduce_sum(tf.pow(current_vstates, 2), axis=-1))
        current_intensities = self.intensity_obj.apply_transfer_function(current_norms)

        assert upper_intensity_bound >= tf.reduce_sum(current_intensities) #TODO remove after a while

        random_val = tf.random.uniform([])
        if random_val * upper_intensity_bound > tf.reduce_sum(current_intensities):
            return False

        current_sender = None  #TODO remove after a while
        for vertex_id, vertex_int_boundary in enumerate(tf.cumsum(current_intensities)):
            if random_val * upper_intensity_bound <= vertex_int_boundary:
                current_sender = vertex_id
                break
        assert current_sender is not None  #TODO remove after a while

        vstates_mask = tf.one_hot(
            indices=current_sender,
            depth=self.intensity_obj.num_vertices,
            on_value=False,
            off_value=True
        )
        
        sender_vstate = current_vstates[current_sender]
        sender_norm = current_norms[current_sender]

        other_vstates = tf.boolean_mask(current_vstates, vstates_mask)
        other_norms = tf.boolean_mask(current_norms, vstates_mask)

        cos_sims = tf.einsum('ij,j->i', other_vstates, sender_vstate) / (sender_norm * other_norms)
        softmax_sims = tf.math.softmax(cos_sims)

        random_val = tf.random.uniform([])
        for vertex_id, vertex_prob_boundary in enumerate(tf.cumsum(softmax_sims)):
            if random_val <= vertex_prob_boundary:
                current_recipient = vertex_id
                break

        if current_sender <= current_recipient:
            current_recipient += 1

        return current_sender, current_recipient

    def _get_intensity_upper_bound(self, c, c_base, o):
        h1 = tf.squeeze(self.cell._h_func(c, o))
        h2 = tf.squeeze(self.cell._h_func(c_base, o))

        W = self.intensity_obj.W

        res1 = tf.einsum('ijk,k->ijk', W, h1)
        res2 = tf.einsum('ijk,k->ijk', W, h2)

        res_min = tf.minimum(res1, res2)
        res_max = tf.maximum(res1, res2)

        vstate_min = tf.abs(tf.reduce_sum(res_min, axis=-1))
        vstate_max = tf.abs(tf.reduce_sum(res_max, axis=-1))

        vstate_opt = tf.maximum(vstate_min, vstate_max)
        upper_bound = tf.reduce_sum(self.intensity_obj.apply_transfer_function(
            tf.sqrt(tf.reduce_sum(tf.pow(vstate_opt, 2), axis=-1))
        ))

        return upper_bound

    def _parse_type_to_one_hot(self, event_type):
        current_sender, current_recipient = event_type
        return tf.concat(
            [
                tf.one_hot(current_sender + 1, self.intensity_obj.num_vertices + 1),
                tf.one_hot(current_recipient, self.intensity_obj.num_vertices)
            ],
            axis=-1
        )
