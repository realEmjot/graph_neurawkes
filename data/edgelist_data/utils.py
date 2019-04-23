import numpy as np
import pandas as pd
import tensorflow as tf


def _get_pair_id_with_loops(i, j, num_ids):
    return num_ids * i + j


def _get_pair_id_without_loops(i, j, num_ids):
    assert i != j
    return num_ids * i + j - i - int(i < j)


def translate_pair_id_with_loops(pair_id, num_ids):
    return pair_id // num_ids, pair_id % num_ids


def translate_pair_id_without_loops(pair_id, num_ids):
    i = pair_id // (num_ids - 1)
    j = pair_id % (num_ids - 1)
    j += int(j >= i)
    return i, j


def _get_event_ids(df, num_ids, loops=True):
    if loops:
        pair_func = _get_pair_id_with_loops
    else:
        pair_func = _get_pair_id_without_loops

    return np.array([pair_func(i, j, num_ids)
                    for i, j in zip(df.sender, df.recipient)],
                    dtype=np.int32)


def _get_df(filepath):
    df = pd.read_csv(filepath, names=['sender', 'recipient', 'time'],
                       dtype={'sender': np.int32, 'recipient': np.int32,
                              'time': np.float32})
    df.time -= df.time.min()
    df.time += 1

    return df


def cut_evenly(df, piece_len, min_len=2, take_rest=False):
    if piece_len > len(df):
        raise ValueError('piece_len cannot be larger than length of dataset!')

    dfs = [df[start:end].copy() for start, end in
            zip(
                range(0, len(df), piece_len),
                range(piece_len, len(df) + piece_len, piece_len)
            )]

    for df_slice in dfs:
        df_slice.time -= df_slice.time.iloc[0]
        df_slice.time += 1

    if not take_rest or len(dfs[-1]) < min_len:
        dfs = dfs[:-1]

    return dfs


def cut_on_big_gaps(df, min_gap_size, min_len=2):
    t_diffs = np.array([t2 - t1 for t1, t2 in zip(df.time[:-1], df.time[1:])])
    boundaries = np.nonzero(t_diffs >= min_gap_size)[0] + 1

    dfs = []
    for start, end in zip(np.insert(boundaries, 0, 0), np.append(boundaries, len(df))):
        if end - start >= min_len:
            df_slice = df[start:end].copy()
            df_slice.time -= df_slice.time.iloc[0]
            df_slice.time += 1
            dfs.append(df_slice)
    return dfs


def to_event_dataset_naive(filepath, cut_func=None, loops=True, **cut_kwargs):
    df = _get_df(filepath)
    num_ids = max(df.sender.max(), df.recipient.max()) + 1

    if cut_func:
        dfs = cut_func(df, **cut_kwargs)
        data = [(_get_event_ids(df, num_ids, loops), df.time, df.time.max()) for df in dfs]
        ds = tf.data.Dataset.from_generator(lambda: data, (tf.int32, tf.float32, tf.float32))
        return ds, len(dfs)

    ds = tf.data.Dataset.from_tensors((_get_event_ids(df, num_ids, loops), df.time, df.time.max()))
    return ds, 1


def to_event_dataset_sender_only(filepath, cut_func=None, **cut_kwargs):
    df = _get_df(filepath)
    if cut_func:
        dfs = cut_func(df, **cut_kwargs)
        data = [(df.sender, df.time, df.time.max()) for df in dfs]
        ds = tf.data.Dataset.from_generator(lambda: data, (tf.int32, tf.float32, tf.float32))
        return ds, len(dfs)

    ds = tf.data.Dataset.from_tensors((df.sender, df.time, df.time.max()))
    return ds, 1


def to_event_dataset_full(filepath, cut_func=None, **cut_kwargs):
    df = _get_df(filepath)
    if cut_func:
        dfs = cut_func(df, **cut_kwargs)
        data = [(df.sender, df.recipient, df.time, df.time.max()) for df in dfs]
        ds = tf.data.Dataset.from_generator(lambda: data, (tf.int32, tf.int32, tf.float32, tf.float32))
        return ds, len(dfs)

    ds = tf.data.Dataset.from_tensors((df.sender, df.recipient, df.time, df.time.max()))
    return ds, 1
