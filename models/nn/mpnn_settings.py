"""
Deafault MPNNs settings
"""


gin = {
    'act_function': 'relu',
    'eps': 0,
    'train_eps': True
}

gat = {
    'gat_version': 1,
    'num_heads': 4,
    'concat': True,
    'negative_slope': 0.2,
    'add_self_loops': True
}

gcn = {
    'improved': True,
    'cached': False,
    'add_self_loops': True,
    'normalize_in': True,
    'bias': True
}

egnn = {
    'velocity': False,
    'cosine_cutoff': True,
    'n_rbf': 1,
    'cutoff': 1

}

MPNN_DEFAULT = {
    'gin': gin,
    'gat': gat,
    'gcn': gcn,
    'egnn': egnn
}