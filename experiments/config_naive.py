from base_config import BASE_RUN

BASE_NAIVE_RUN = {
    **BASE_RUN,
    'default_values': {
        **BASE_RUN['default_values'],
        'data_mode': 'naive',
        'batching_mode': 'gap_cut',
        'loops':  False
    },
    'run': 'default'
}

RUN_naive_basic_fb = {
    **BASE_NAIVE_RUN,
    'default_values': {
        **BASE_NAIVE_RUN['default_values'],
        'data_path': '../data/edgelist_data/fb-forum/data.txt',
        'num_types': 899,
        'model_savepath': 'naive_saved_models/naive_basic_fb/${id}',
        'batching_kwargs': {'min_gap_size': 5000}
    }
}

RUN_naive_basic_ia = {
    **BASE_NAIVE_RUN,
    'default_values': {
        **BASE_NAIVE_RUN['default_values'],
        'data_path': '../data/edgelist_data/ia-contacts_hypertext2009/data.txt',
        'num_types': 113,
        'model_savepath': 'naive_saved_models/naive_basic_ia/${id}',
        'batching_kwargs': {'min_gap_size': 75}
    }
}