## Graph Neural Hawkes and Neural Hawkes Tensorflow implementation

### Basic Usage

```bash
python app.py mode config
```

`mode` must be one of *train* or *generate*.  
`config` must be a valid path to config in *.json* format.

### Config

A config file must be in *.json* format, and contain a dictionary with settings, like in an `example_config.json`.  

#### Train config keys

- model_mode: one of *"GNH"* (Graph Neural Hawkes) or *"NH"* (Neural Hawkes)
- results_savepath: training statistics savepath
- model_savepath: model learned weghts savepath 
- data_path: path to a valid text file containing input event stream network (examples are contained in `data` directory)
- self_links
- directed
- num_epochs
- batch_size
- num_units
- num_types
- N_ratio
- vstate_len: only in case of *GNH* model mode
- batching_mode: one of *"gap_cut"*, *"even_cut"* or *None*
- batching_kwargs: a dictionary containing batching settings

##### Batching settings

For *gap_cut*:
- min_gap_size
- [min_len]

For *even_cut*:
- piece_len
- [min_len]
- [take_rest]


#### Generation config keys

- model_mode: one of *GNH* (Graph Neural Hawkes) or *NH* (Neural Hawkes)
- results_savepath
- num_units
- num_types
- vstate_len
- self_links
- model_savepath: 
- seed
- max_events
- max_time
