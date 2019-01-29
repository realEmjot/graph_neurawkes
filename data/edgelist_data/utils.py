import numpy as np
import pandas as pd
import tensorflow as tf


def _get_pair_id(i, j, num_ids):
    return 10 * i + j

def _get_df(filepath):
    df = pd.read_csv(filepath, names=['sender', 'recipient', 'time'],
                       dtype={'sender': np.int32, 'recipient': np.int32,
                              'time': np.float32})
    df.sender -= 1
    df.recipient -= 1
    df.time -= df.time.min()
    df.time += 1

    return df

def to_event_dataset_naive(filepath):
    df = _get_df(filepath)
    event_ids = np.array([_get_pair_id(i, j, max(df.sender.max(), df.recipient.max()) + 1)
                          for i, j in zip(df.sender, df.recipient)],
                          dtype=np.int32)

    ds = tf.data.Dataset.from_tensors((event_ids, df.time, df.time.max()))
    return ds

def to_event_dataset_sender_only(filepath):
    df = _get_df(filepath)
    ds = tf.data.Dataset.from_tensors((df.sender, df.time, df.time.max()))
    return ds
