from typing import Literal


import torch
from torch.nn import (Linear,
                      Sequential,
                      ReLU, 
                      SiLU,
                      Sigmoid,
                      Tanh)


from torch_geometric.nn import (BatchNorm,
                                GraphNorm,
                                LayerNorm,
                                global_add_pool,
                                global_mean_pool,
                                global_max_pool)


from torch_geometric.utils import dropout_edge


from torch.nn import Module





def make_act_function(function_name: Literal['relu', 'silu', 'sigmoid', 'tanh']):

    act_functions = {'relu': ReLU,
                     'silu': SiLU,
                     'sigmoid': Sigmoid,
                     'tanh': Tanh}
    
    return act_functions[function_name]



def make_normalization(name: Literal['batchnorm', 'graphnorm', 'layernorm']):

    norm_functions = {'batchnorm': BatchNorm,
                      'graphnorm': GraphNorm,
                      'layernorm': LayerNorm}
    
    return norm_functions[name]



def make_pooling(glob_pool: Literal['sum',
                                    'mean',
                                    'max']):

    if glob_pool == 'sum':
        return global_add_pool

    elif glob_pool == 'mean':
        return global_mean_pool

    elif glob_pool == 'max':
        return global_max_pool


def distance(xi, xj) -> torch.Tensor:

    r_norm = torch.norm(xi - xj, dim=-1)
    r_norm = r_norm **2
    return r_norm.unsqueeze(-1)


def cosine_cutoff(r, cutoff):
    return 0.5 * (torch.cos(torch.pi * r / cutoff) + 1.0) * (r < cutoff).float()


def mlp(in_dim, 
        hidden_dim, 
        out_dim,
        act_function='relu'):
    
    return Sequential(
        Linear(in_dim, hidden_dim),
        make_act_function(act_function)(),
        Linear(hidden_dim, hidden_dim),
        make_act_function(act_function)(),
        Linear(hidden_dim, out_dim)
        )



class DropoutEdge(Module):

    def __init__(self, dropout_p, undirected=True):
        super().__init__()

        self.dropout_p = dropout_p
        self.undirected = undirected

    def forward(self, edge_index):
        return dropout_edge(edge_index, self.dropout_p, force_undirected=self.undirected, training=self.training)



class RBF(Module):

    def __init__(self, n_rbf,cutoff=10):
        super().__init__()

        self.n_rbf = n_rbf
        centers = torch.linspace(0, cutoff, n_rbf, dtype=torch.float32)
        self.gamma = cutoff
        self.register_buffer('centers', centers)

    def forward(self, norm):

        rbf = torch.exp(-self.gamma * (norm - self.centers) ** 2)
        return rbf