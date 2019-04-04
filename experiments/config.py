from base_config import BASE_RUN


RUN_full_gap_fb = {
    **BASE_RUN,
    'default_values': {
        **BASE_RUN['default_values'],
        'batching_mode': 'gap_cut',
        'data_path': '../data/edgelist_data/fb-forum/data.txt',
        'num_types': 899,
        'model_savepath': 'saved_models/full_gap_fb/${id}'
    },
    'values': {
        **BASE_RUN['values'],
        'batching_kwargs': [
            {'min_gap_size': 1000},
            {'min_gap_size': 2500},
            {'min_gap_size': 5000}
        ]
    },
    'run': {
        'individual_search': {
            'args': ['batching_kwargs'],
            'individual_search': ['num_units', 'vstate_len']
        }
    }
}


RUN_full_gap_ia = {
    **BASE_RUN,
    'default_values': {
        **BASE_RUN['default_values'],
        'batching_mode': 'gap_cut',
        'data_path': '../data/edgelist_data/ia-contacts_hypertext2009/data.txt',
        'num_types': 113,
        'model_savepath': 'saved_models/full_gap_ia/${id}'
    },
    'values': {
        **BASE_RUN['values'],
        'batching_kwargs': [
            {'min_gap_size': 50},
            {'min_gap_size': 75},
            {'min_gap_size': 100}
        ]
    },
    'run': {
        'individual_search': {
            'args': ['batching_kwargs'],
            'individual_search': ['num_units', 'vstate_len']
        }
    }
}

RUN_full_even = {
    **BASE_RUN,
    'default_values': {
        **BASE_RUN['default_values'],
        'batching_mode': 'even_cut'
    },
    'values': {
        **BASE_RUN['values'],
        'batching_kwargs': [
            {'piece_len': 100},
            {'piece_len': 250},
            {'piece_len': 500}
        ]
    },
    'run': {
        'individual_search': {
            'args': [('data_path', 'num_types')],
            'individual_search': {
                'args': ['batching_kwargs'],
                'individual_search': ['num_units', 'vstate_len']
            }
        }
    }
}
