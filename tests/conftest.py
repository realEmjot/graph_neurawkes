import pytest
import numpy as np
import tensorflow as tf


@pytest.fixture
def tf_session():
    with tf.Session() as sess:
        yield sess

@pytest.fixture(params=[1, 234, 5436764])
def random_seed(request):
    np.random.seed(request.param)
