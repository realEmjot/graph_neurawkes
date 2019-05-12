from base_config import BASE_RUN

BASE_NAIVE_RUN = {
    **BASE_RUN,
    'default_values': {
        **BASE_RUN['default_values'],
        'data_mode': 'naive'
    },
    'run': 'default'
}

RUN_naive_basic_fb = {
    **BASE_NAIVE_RUN,
    'default_values': {
        **BASE_NAIVE_RUN['default_values'],
        'data_path': '../data/edgelist_data/fb-forum/data.csv',
        'self_links': True,
        'num_types': 899,
        'model_savepath': 'naive_saved_models/naive_basic_fb',
        'batching_kwargs': {'min_gap_size': 5000},
        'num_units': 64,
        'vstate_len': 50
    }
}

RUN_naive_basic_ia = {
    **BASE_NAIVE_RUN,
    'default_values': {
        **BASE_NAIVE_RUN['default_values'],
        'data_path': '../data/edgelist_data/ia-contacts_hypertext2009/data.csv',
        'self_links': False,
        'num_types': 113,
        'model_savepath': 'naive_saved_models/naive_basic_ia',
        'batching_kwargs': {'min_gap_size': 75},
        'num_units': 64,
        'vstate_len': 50
    }
}

RUN_naive_basic_radoslaw = {
    **BASE_NAIVE_RUN,
    'default_values': {
        **BASE_NAIVE_RUN['default_values'],
        'data_path': '../data/edgelist_data/ia-radoslaw-email/data.csv',
        'self_links': True,
        'num_types': 167,
        'model_savepath': 'naive_saved_models/naive_basic_radoslaw',
        'batching_kwargs': {'min_gap_size': 45000},
        'num_units': 64,
        'vstate_len': 50
    }
}
