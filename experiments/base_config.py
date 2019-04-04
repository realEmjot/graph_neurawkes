import tensorflow as tf

import sys
sys.path.insert(0, '..')

from graph_neurawkes import train_and_save


BASE_RUN = {
    'function': train_and_save,
    'default_values': {
        'num_epochs': 10,
        'batch_size': 1,
        'num_units': 128,
        'vstate_len': 100,
        'val_ratio': 0.2,
        'data_type': 'edgelist',
        'data_mode': 'full'
    },
    'values': {
        'data_path': [
            '../data/edgelist_data/fb-forum/data.txt',
            '../data/edgelist_data/ia-contacts_hypertext2009/data.txt'   
        ],
        'num_types': [899, 113],
        'num_units': [64, 256, 512],
        'vstate_len': [50, 200, 300]
    },
    'post_hook': tf.reset_default_graph
}
