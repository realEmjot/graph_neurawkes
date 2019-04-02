def sample_function(**kwargs):
    print(kwargs)
    return 5

def simple_hook():
    pass


BASE_RUN = {
    'function': sample_function,
    'default_values': {
        'num_epochs': 10,
        'batch_size': 1,
        'num_units': 128,
        'vstate_len': 100,
        'val_ratio': 0.2,
        'data_type': 'edgelist',
        'data_mode': 'full'
    },
    'values': {
        'data_path': [
            '../data/edgelist_data/fb-forum/data.txt',
            '../data/edgelist_data/ia-contacts_hypertext2009/data.txt'   
        ],
        'num_types': [899, 113],
        'num_units': [64, 256, 512],
        'vstate_len': [50, 200, 300]
    },
    'run': {
        'individual_search': ['num_units', 'vstate_len']
    },
    'post_hook': simple_hook
}


RUN_1 = {
    **BASE_RUN,
    'default_values': {
        **BASE_RUN['default_values'],
        'batching_mode': 'gap_cut'
    },
    'values': {
        **BASE_RUN['values'],
        'batching_kwargs': [
            {'min_gap_size': 10},
            {'min_gap_size': 20}
        ]
    },
    'run': {
        'individual_search': {
            'args': [('data_path', 'num_types', 'batching_kwargs')],
            'individual_search': ['num_units', 'vstate_len']
        }
    }
}


RUN_2 = {
    **BASE_RUN,
    'default_values': {
        **BASE_RUN['default_values'],
        'batching_mode': 'even_cut'
    },
    'values': {
        **BASE_RUN['values'],
        'batching_kwargs': [
            {'piece_len': 10},
            {'piece_len': 100}
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
