import numpy as np
import pandas as pd
import tensorflow as tf


def _get_pair_id(i, j, num_ids):
    return 10 * i + j

def _get_df(filepath):
    df = pd.read_csv(filepath, names=['sender', 'recipient', 'time'],
                       dtype={'time': np.float32})
    df.sender -= 1
    df.recipient -= 1
    df.time -= df.time.min()
    df.time += 1

    df.sender = df.sender.astype(np.int32, copy=False)
    df.recipient = df.recipient.astype(np.int32, copy=False)

    return df

def _cut_df(df, difference_cut, min_len=2):
    t_diffs = np.array([t2 - t1 for t1, t2 in zip(df.time[:-1], df.time[1:])])
    boundaries = np.nonzero(t_diffs >= difference_cut)[0] + 1

    dfs = []
    for start, end in zip(np.insert(boundaries, 0, 0), np.append(boundaries, len(df))):
        if end - start >= min_len:
            df_slice = df[start:end]
            df_slice.time -= df_slice.time.iloc[0]
            df_slice.time += 1
            dfs.append(df_slice)
    return dfs

def to_event_dataset_naive(filepath):
    df = _get_df(filepath)
    event_ids = np.array([_get_pair_id(i, j, max(df.sender.max(), df.recipient.max()) + 1)
                          for i, j in zip(df.sender, df.recipient)],
                          dtype=np.int32)

    ds = tf.data.Dataset.from_tensors((event_ids, df.time, df.time.max()))
    return ds, 1

def to_event_dataset_sender_only(filepath, difference_cut=None):
    df = _get_df(filepath)
    if difference_cut:
        dfs = _cut_df(df, difference_cut)
        data = [(df.sender, df.time, df.time.max()) for df in dfs]
        ds = tf.data.Dataset.from_generator(lambda: data, (tf.int32, tf.float32, tf.float32))
        return ds, len(dfs)

    ds = tf.data.Dataset.from_tensors((df.sender, df.time, df.time.max()))
    return ds, 1
