

# cont_lstm.py
## class ContLSTMCell
### function \_\_init\_\_
[self.] num_units: scalar  
[self.] elem_size: scalar  

[self.] .+(_base)?_vars:  
* W: [elem_size x num_units]  
* U: [num_units x num_units]  
* d: [num_units]  

### function build_graph
x: [batch_size x elem_size]

i, f, z, o, i_base, f_base, decay: [batch_size x num_units]  
c, c_base: [batch_size x num_units]  

### function get_time_dependent_vars
t: [batch_size]  
graph_vars:
* t_init: [batch_size]  
* c, c_base, decay, o: [batch_size x num_units]  

c_t, h_t: [batch_size x num_units]  

### function _create_gate
W: [elem_size x num_units]  
U: [num_units x num_units]  
d: [num_units]  
x: [batch_size x elem_size]  
h_init: [batch_size x num_units]  

*returns*: [batch_size x num_units]  


# intensity.py
## class GraphIntensity
### function \_\_init\_\_
num_units: scalar  
[self.] num_vertices: scalar  
vstate_len: scalar  

self.W: [num_vertices x vstate_len x num_units]  
self.s: [num_vertices]  

### function get_intensity_for_pairs_sequence
x1_seq: [seq_len]  
x2_seq: [seq_len]  
cstm_hs: [seq_len x num_units]  

W1: [seq_len x vstate_len x num_units]  
s: [seq_len]  

v1_states: [seq_len x vstate_len]  
all_states: [seq_len x num_vertices x v_state_len]  

indices: [num_vertices]  
all_states: [seq_len x num_vertices-1 x v_state_len]  

dot_products: [seq_len x num_vertices-1]  
v1_norms: [seq_len]  
all_norms: [seq_len x num_vertices-1]  
multiplied_norms: [seq_len x num_vertices-1]  

cos_sims: [seq_len x num_vertices-1]  
softmax_sims: [seq_len x num_vertices-1]  

v2_indices: [seq_len]  
v2_indices_mask: [seq_len x num_vertices-1]  
v2_sims: [seq_len]  

*returns*: [seq_len]  

### function get_intensity_for_every_vertex_in_batchseq
ctsm_Hs: [batch_size x N x num_units]  

all_states: [batch_size x N x num_vertices x vstate_len]  
all_norms: [batch_size x N x num_vertices]   

### function apply_transfer_function
input_: [? x num_vertices] 
