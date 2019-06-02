import functools
import random

import numpy as np
import pandas as pd
import tensorflow as tf

SHUFFLE_SEED = 0.1995


def _get_pair_id_with_self_links(i, j, num_ids):
    return num_ids * i + j


def _get_pair_id_without_self_links(i, j, num_ids):
    assert i != j
    return num_ids * i + j - i - int(i < j)


def _get_pair_id_with_self_links_undir(i, j, num_ids):
    return num_ids * i + j - i * (i + 1) / 2


def _get_pair_id_without_self_links_undir(i, j, num_ids):
    assert i != j
    return num_ids * i + j - i * (i + 1) / 2 - i - 1


def _translate_pair_id_with_self_links(pair_id, num_ids):
    return pair_id // num_ids, pair_id % num_ids


def _translate_pair_id_without_self_links(pair_id, num_ids):
    i = pair_id // (num_ids - 1)
    j = pair_id % (num_ids - 1)
    j += int(j >= i)
    return i, j


def translate_pair_id(pair_id, num_ids, self_links):
    if self_links:
        return _translate_pair_id_with_self_links(pair_id, num_ids)
    else:
        return _translate_pair_id_without_self_links(pair_id, num_ids)


def _get_event_ids(df, num_ids, self_links=True, directed=True):
    if self_links:
        if directed:
            pair_func = _get_pair_id_with_self_links
        else:
            pair_func = _get_pair_id_with_self_links_undir
    else:
        if directed:
            pair_func = _get_pair_id_without_self_links
        else:
            pair_func = _get_pair_id_without_self_links_undir

    return np.array([pair_func(i, j, num_ids)
                    for i, j in zip(df.sender, df.recipient)],
                    dtype=np.int32)


def _get_df_from_csv(filepath, subtract_min_time=True):
    df = pd.read_csv(filepath, names=['sender', 'recipient', 'time'],
                       dtype={'sender': np.int32, 'recipient': np.int32,
                              'time': np.float32})
    if subtract_min_time:
        df.time -= df.time.min()
        df.time += 1

    return df


def _get_df_from_sequence(seq, subtract_min_time=False):
    df = pd.DataFrame(seq, columns=['sender', 'recipient', 'time'])

    df.sender = df.sender.astype(np.int32)
    df.recipient = df.recipient.astype(np.int32)
    df.time = df.time.astype(np.float32)

    if subtract_min_time:
        df.time -= df.time.min()
        df.time += 1

    return df


def _cut_func_shuffle_decorator(cut_func):
    @functools.wraps(cut_func)
    def wrapper(*args, shuffle=True, **kwargs):
        dfs = cut_func(*args, **kwargs)
        if shuffle:
            random.shuffle(dfs, lambda: SHUFFLE_SEED)
        return dfs
    return wrapper


@_cut_func_shuffle_decorator
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


@_cut_func_shuffle_decorator
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


def to_event_dataset_naive(filepath=None, seq=None, seqs=None, num_ids=None, max_times=None, cut_func=None, self_links=True, directed=True, **cut_kwargs):
    if filepath is not None:
        df = _get_df_from_csv(filepath)
        num_ids = max(df.sender.max(), df.recipient.max()) + 1
    elif seq is not None:
        df = _get_df_from_sequence(seq)
    else:
        dfs = [_get_df_from_sequence(seq) for seq in seqs]

    if cut_func or seqs:
        if not seqs:
            dfs = cut_func(df, **cut_kwargs)
        if max_times is None:
            data = [(_get_event_ids(df, num_ids, self_links, directed), df.time, df.time.max()) for df in dfs]
        else:
            assert len(dfs) == len(max_times)
            # TODO check if every max time is larger than last event time
            data = [(_get_event_ids(df, num_ids, self_links, directed), df.time, max_time) for df, max_time in zip(dfs, max_times)]
        ds = tf.data.Dataset.from_generator(lambda: data, (tf.int32, tf.float32, tf.float32))
        return ds, len(dfs)

    ds = tf.data.Dataset.from_tensors((_get_event_ids(df, num_ids, self_links, directed), df.time, df.time.max()))
    return ds, 1


def to_event_dataset_sender_only(filepath=None, seq=None, seqs=None, max_times=None, cut_func=None, **cut_kwargs):
    if filepath is not None:
        df = _get_df_from_csv(filepath)
    elif seq is not None:
        df = _get_df_from_sequence(seq)
    else:
        dfs = [_get_df_from_sequence(seq) for seq in seqs]

    if cut_func or seqs:
        if not seqs:
            dfs = cut_func(df, **cut_kwargs)
        if max_times is None:
            data = [(df.sender, df.time, df.time.max()) for df in dfs]
        else:
            assert len(dfs) == len(max_times)
            # TODO check if every max time is larger than last event time
            data = [(df.sender, df.time, max_time) for df, max_time in zip(dfs, max_times)]
        ds = tf.data.Dataset.from_generator(lambda: data, (tf.int32, tf.float32, tf.float32))
        return ds, len(dfs)

    ds = tf.data.Dataset.from_tensors((df.sender, df.time, df.time.max()))
    return ds, 1


def to_event_dataset_full(filepath=None, seq=None, seqs=None, max_times=None, cut_func=None, **cut_kwargs):
    if filepath is not None:
        df = _get_df_from_csv(filepath)
    elif seq is not None:
        df = _get_df_from_sequence(seq)
    else:
        dfs = [_get_df_from_sequence(seq) for seq in seqs]

    if cut_func or seqs:
        if not seqs:
            dfs = cut_func(df, **cut_kwargs)
        if max_times is None:
            data = [(df.sender, df.recipient, df.time, df.time.max()) for df in dfs]
        else:
            assert len(dfs) == len(max_times)
            # TODO check if every max time is larger than last event time
            data = [(df.sender, df.recipient, df.time, max_time) for df, max_time in zip(dfs, max_times)]
        ds = tf.data.Dataset.from_generator(lambda: data, (tf.int32, tf.int32, tf.float32, tf.float32))
        return ds, len(dfs)

    ds = tf.data.Dataset.from_tensors((df.sender, df.recipient, df.time, df.time.max()))
    return ds, 1
