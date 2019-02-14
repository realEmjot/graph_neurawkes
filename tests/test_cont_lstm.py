import pytest
import numpy as np
import tensorflow as tf

from my_neurawkes.cont_lstm import ContLSTMCell, ContLSTMTrainer


@pytest.fixture(scope='module', params=[64, 128, 256])
def cstm_cell(request):
    yield ContLSTMCell(request.param, 10)
    tf.reset_default_graph()

@pytest.fixture(scope='module')
def cstm_trainer(cstm_cell):
    return ContLSTMTrainer(cstm_cell)

@pytest.fixture(
    scope='module',
    params=[
        (np.array([2, 3, 1, 0]), np.array([[0, 2], [2, 5], [5, 6], [6, 6]])),
        (np.array([0, 0, 4, 5, 1, 6]), np.array([[0, 0], [0, 0], [0, 4], [4, 9], [9, 10], [10, 16]]))
    ]
)
def counts_data(request):
    yield request.param


@pytest.mark.usefixtures('random_seed')
def test_get_inter_t_mask(tf_session, cstm_trainer):
    for _ in range(10):
        T_max = np.random.uniform(1, 1000)
        t_seq = np.random.uniform(0, T_max, size=(2, 5))
        inter_t_seq = np.random.uniform(1e-10, T_max, size=(2, 10))

        res_tf = cstm_trainer._get_inter_t_mask(t_seq, inter_t_seq, T_max)
        res_tf = tf_session.run(res_tf)

        seq1 = np.pad(t_seq, [[0, 0], [1, 0]], mode='constant', constant_values=0)
        seq2 = np.pad(t_seq, [[0, 0], [0, 1]], mode='constant', constant_values=T_max)

        res_np = (seq1[:, np.newaxis] < inter_t_seq[..., np.newaxis]) & \
                 (seq2[:, np.newaxis] >= inter_t_seq[..., np.newaxis])
        res_np = np.transpose(res_np, (2, 0, 1))  # batch x N x S -> S x batch x N

        assert np.array_equal(res_tf, res_np)

def test_convert_counts_to_slices(tf_session, cstm_trainer, counts_data):
    TEST_DATA, TEST_RES = counts_data

    res = cstm_trainer._convert_counts_to_slices(TEST_DATA)
    res = tf_session.run(res)

    assert np.array_equal(res, TEST_RES)