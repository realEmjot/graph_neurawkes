import tensorflow as tf
from my_neurawkes import Neurawkes
from data.edgelist_data.utils import to_event_dataset_sender_only, to_event_dataset_naive
from data.neurawkes_data.utils import to_dataset


def run_experiment_edge(data_path,
        num_epochs, batch_size, num_units,
        data_label='train'):
    data_fb, data_size = to_event_dataset_sender_only(data_path, 5000)
    model = Neurawkes(num_units, 899)

    with tf.Session() as sess:
        return model.train(sess, data_fb, 1., num_epochs, batch_size, data_size, 0.2)

def run_experiment_edge_ultra(data_path,
        num_epochs, batch_size, num_units,
        data_label='train'):
    data_fb, data_size = to_event_dataset_naive(data_path, 5000)
    model = Neurawkes(num_units, 808201)

    with tf.Session() as sess:
        return model.train(sess, data_fb, 1., num_epochs, batch_size, data_size, 0.2)

def run_experiment_neurawkes(data_path,
        num_epochs, batch_size, num_units,
        data_label='train'):
    data_fb, data_size = to_dataset(data_path, data_label)
    model = Neurawkes(num_units, 3)

    with tf.Session() as sess:
        return model.train(sess, data_fb, 1., num_epochs, batch_size, data_size, 0.2)

# run_experiment_neurawkes('data/neurawkes_data/data_retweet', 10, 10, 64)