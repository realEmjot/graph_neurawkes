import tensorflow as tf
from my_neurawkes import Neurawkes
from data.edgelist_data.utils import to_event_dataset_sender_only


def run_experiment(data_path, data_label='train'):
    data_fb, data_size = to_event_dataset_sender_only(data_path, 5000)
    model = Neurawkes(128, 899)

    with tf.Session() as sess:
        model.train(sess, data_fb, 1., 1, 1, data_size)

if __name__ == '__main__':
    run_experiment('data/edgelist_data/fb-forum.txt')
