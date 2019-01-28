import tensorflow as tf
from my_neurawkes import Neurawkes
from data.utils import to_dataset


data_retweets = to_dataset('data/data_retweet', 'train')
model = Neurawkes(100, 3)

with tf.Session() as sess:
    model.train(sess, data_retweets, 1., 1000, 10, 20000)
