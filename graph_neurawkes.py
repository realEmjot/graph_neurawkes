import functools
import os
import pickle

import tensorflow as tf

import data.edgelist_data.utils as edgelist_utils
from src import Neurawkes, GraphNeurawkes


_mode = None  # global on purpose!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _check_mode_decorator(expected_mode):
    global _mode

    if _mode is None:
        _mode = expected_mode
    elif _mode != expected_mode:
        raise RuntimeError('You cannot both train the model and generate'
                            'from it using the same python session :(')

def train_and_save(
        data_path,
        num_epochs, batch_size,
        num_units, num_types,
        N_ratio=1.0,
        vstate_len=None,
        val_ratio=None,
        results_savepath=None, model_savepath=None,
        data_type='edgelist',
        data_mode='full',
        batching_mode=None,
        batching_kwargs=None,
        self_links=False):

    _check_mode_decorator('static')

    if data_type == 'edgelist':
        if batching_mode is None:
            cut_func=None
        else:
            if batching_mode == 'even_cut':
                cut_func = edgelist_utils.cut_evenly
            elif batching_mode == 'gap_cut':
                cut_func = edgelist_utils.cut_on_big_gaps
            else:
                raise NotImplementedError

            if batching_kwargs is None:
                batching_kwargs = {}
            else:
                pass # TODO make this somehow inputable from command line
        
        additional_kwargs = {}
        if data_mode == 'full':
            ds_func = edgelist_utils.to_event_dataset_full
            model_class = GraphNeurawkes
        elif data_mode == 'sender_only':
            ds_func = edgelist_utils.to_event_dataset_sender_only
            model_class = Neurawkes
        elif data_mode == 'naive':
            ds_func = edgelist_utils.to_event_dataset_naive
            model_class = Neurawkes
            additional_kwargs = {'self_links': self_links}

            if self_links:
                num_types = pow(num_types, 2)
            else:
                num_types = pow(num_types, 2) - num_types

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if model_class is Neurawkes:
        model = Neurawkes(num_units, num_types)
    if model_class is GraphNeurawkes:
        if vstate_len is None:
            raise ValueError
        model = GraphNeurawkes(num_units, num_types, vstate_len, self_links)

    dataset, dataset_size = ds_func(data_path, cut_func, **additional_kwargs,
                                    **batching_kwargs)

    with tf.Session() as sess:
        result = model.train(
            sess, dataset, N_ratio, num_epochs, batch_size, dataset_size,
            val_ratio, model_savepath
        )

    if results_savepath:
        with open(results_savepath, 'wb') as f:
            pickle.dump(result, f)

    return result


def generate():
    _check_mode_decorator('eager')
    pass
    


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        prog='Dynamic network generator',
        description='Generates dynamic networks, using Graph Neurawkes model. '
                     'Vanilla Neurawkes model is included as well'
    )

    args = parser.parse_args()

    # TODO