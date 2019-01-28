import tensorflow as tf
from my_neurawkes import Neurawkes
from data.utils import to_dataset


def test(data_path, data_label='train'):
    data_retweets = to_dataset(data_path, data_label)
    model = Neurawkes(100, 3)

    with tf.Session() as sess:
        model.train(sess, data_retweets, 1., 1000, 10, 20000)

if __name__ == '__main__':
    test('data/data_retweet')
