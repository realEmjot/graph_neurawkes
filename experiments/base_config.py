import tensorflow as tf

import sys
sys.path.insert(0, '..')

from graph_neurawkes import train_and_save


BASE_RUN = {
    'function': train_and_save,
    'default_values': {
        'num_epochs': 30,
        'batch_size': 1,
        'val_ratio': 0.2,
        'data_type': 'edgelist',
        'data_mode': 'full'
    },
    'post_hook': tf.reset_default_graph
}
