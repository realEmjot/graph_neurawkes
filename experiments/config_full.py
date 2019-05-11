from base_config import BASE_RUN


RUN_full = {
    **BASE_RUN,
    'default_values': {
        **BASE_RUN['default_values'],
        'model_savepath': 'full_saved_models/${id}'
    },
    'values': {
        'data_path': [
            '../data/edgelist_data/fb-forum/data.csv',
            '../data/edgelist_data/ia-contacts_hypertext2009/data.csv',
            '../data/edgelist_data/ia-radoslaw-email/data.csv'
        ],
        'self_links': [True, False, True],
        'num_types': [899, 113, 167],
        'batching_kwargs': [
            {'min_gap_size': 5000},
            {'min_gap_size': 75},
            {'min_gap_size': 45000}
        ],

        'num_units': [64, 128, 256],
        'vstate_len': [50, 100, 200, 300]
    },
    'run': {
        'individual_search': {
            'args': [('data_path', 'self_links', 'num_types', 'batching_kwargs')],
            'grid_search': ['num_units', 'vstate_len']
        }
    }
}
