import os.path
import pickle
import tensorflow as tf


def to_dataset(filepath, type_):
    with open(os.path.join(filepath, type_) + '.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    data = [([int(e['type_event']) for e in seq], [e['time_since_start'] for e in seq])
            for seq in data[type_]]
    print(len(data))

    # def gen():
    #     for seq in data[type_]:
    #         yield [int(e['type_event']) for e in seq], [e['time_since_start'] for e in seq]

    ds = tf.data.Dataset.from_generator(lambda: data, (tf.int32, tf.float32))
    ds = ds.map(lambda x, y: (x, y, y[-1]))

    return ds
