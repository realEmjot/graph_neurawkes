import functools
import json
import os
import pickle

import tensorflow as tf
from scipy.special import binom

import data.edgelist_data.utils as edgelist_utils
from src import Neurawkes, GraphNeurawkes


_mode = None  # global on purpose!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _check_mode(expected_mode):
    global _mode

    if _mode is None:
        _mode = expected_mode
    elif _mode != expected_mode:
        raise RuntimeError('You cannot both train the model and generate'
                            'from it using the same python session!')



def train_and_save(
        *,
        data_path,
        num_epochs, batch_size,
        num_units, num_types,
        N_ratio=1.0,
        vstate_len=None,
        val_ratio=None,
        results_savepath=None, model_savepath=None,
        model_mode='GNH',
        batching_mode=None,
        batching_kwargs=None,
        self_links=True,
        directed=True):

    _check_mode('static')

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
    
    additional_kwargs = {}
    if model_mode == 'GNH':
        ds_func = edgelist_utils.to_event_dataset_full
        model_class = GraphNeurawkes
    elif model_mode == 'NH':
        ds_func = edgelist_utils.to_event_dataset_naive
        model_class = Neurawkes
        additional_kwargs = {
            'self_links': self_links,
            'directed': directed
        }

        if directed:
            new_num_types = pow(num_types, 2)
        else:
            new_num_types = int(binom(num_types, 2)) + num_types

        if not self_links:
            new_num_types -= num_types

        num_types = new_num_types

    else:
        raise NotImplementedError

    if model_class is Neurawkes:
        model = Neurawkes(num_units, num_types)
    if model_class is GraphNeurawkes:
        if vstate_len is None:
            raise ValueError
        model = GraphNeurawkes(num_units, num_types, vstate_len, self_links)

    dataset, dataset_size = ds_func(filepath=data_path, cut_func=cut_func,
                                    **additional_kwargs, **batching_kwargs)

    with tf.Session() as sess:
        result = model.train(
            sess, dataset, N_ratio, num_epochs, batch_size, dataset_size,
            val_ratio, model_savepath
        )

    if results_savepath:
        with open(results_savepath, 'wb') as f:
            pickle.dump(result, f)

    return result


def generate(
    *,
    results_savepath,
    num_units,
    num_types,
    vstate_len=None,
    self_links=True,
    model_mode='GNH',
    **gen_kwargs):

    _check_mode('eager')
    
    if model_mode == 'GNH':
        if vstate_len is None:
            raise ValueError
        model = GraphNeurawkes(num_units, num_types, vstate_len, self_links)
    elif model_mode == 'NH':
        model = Neurawkes(num_units, num_types)
    else:
        raise NotImplementedError
    
    generated_seq = model.generate(**gen_kwargs)
    with open(results_savepath, 'w') as f:
        json.dump(generated_seq, f)



if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog='Dynamic network generator',
        description='Generates dynamic networks, using Graph Neurawkes model. '
                     'Vanilla Neurawkes model is included as well'
    )

    parser.add_argument('mode', choices=['train', 'generate'])
    parser.add_argument('config')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.mode == 'train':
        train_and_save(**config)
    else:
        generate(**config)
