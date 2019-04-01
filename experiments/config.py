def sample_function(**kwargs):
    print(kwargs)
    return 5

RUN_1 = {
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
        'data_path': ['kek', 'lol', 'eh'],
        'num_units': [64, 256, 512],
        'vstate_len': [50, 200, 300]
    },
    'run': {
        'individual_search': {
            'args': ['data_path'],
            'individual_search': ['num_units', 'vstate_len']
        }
    }
}