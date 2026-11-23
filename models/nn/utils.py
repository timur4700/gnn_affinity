from typing import Literal


import torch
from torch.nn import (
    ModuleList,
    Linear,
    LeakyReLU,
    Sequential,
    ReLU, 
    SiLU,
    Sigmoid,
    Tanh,
    Dropout
)


from torch_geometric.nn import (BatchNorm,
                                GraphNorm,
                                LayerNorm,
                                global_add_pool,
                                global_mean_pool,
                                global_max_pool)


from torch_geometric.utils import dropout_edge


from torch.nn import Module



def make_act_function(function_name: Literal['relu', 'silu', 'sigmoid', 'tanh']):

    """Return an **activation function** class by name.

    Args:
        function_name: Name of the activation function.

    Returns:
        The corresponding PyTorch activation module class.
    """

    act_functions = {'relu': ReLU,
                     'leaky_relu': LeakyReLU,
                     'silu': SiLU,
                     'sigmoid': Sigmoid,
                     'tanh': Tanh}
    
    return act_functions[function_name]



def make_normalization(name: Literal['batchnorm', 'graphnorm', 'layernorm']):

    """Return a **normalization layer** class by name.

    Args:
        name: Name of the normalization method. Available Normalizations: BatchNorm, GraphNorm, LayerNorm

    Returns:
        The corresponding normalization module class.
    """

    norm_functions = {'batchnorm': BatchNorm,
                      'graphnorm': GraphNorm,
                      'layernorm': LayerNorm}
    
    return norm_functions[name]



def make_pooling(glob_pool: Literal['sum',
                                    'mean',
                                    'max']):

    """Return a **global graph pooling function** by name.

    Args:
        glob_pool: Pooling operation. Supported options are
            'sum', 'mean', and 'max'.

    Returns:
        The corresponding PyTorch Geometric global pooling function.
    """

    if glob_pool == 'sum':
        return global_add_pool

    elif glob_pool == 'mean':
        return global_mean_pool

    elif glob_pool == 'max':
        return global_max_pool


def distance(xi, xj) -> torch.Tensor:

    """Calculate **squared Euclidean distances** between coordinate vectors.

    Args:
        xi: First set of coordinate vectors.
        xj: Second set of coordinate vectors.

    Returns:
        Squared Euclidean distances with an additional trailing
        dimension of size 1.
    """

    r_norm = torch.norm(xi - xj, dim=-1)
    return r_norm.unsqueeze(-1)


def cosine_cutoff(r, cutoff):

    """Apply a **cosine cutoff envelope** to distance values.

    The envelope smoothly decreases from 1 to 0 as the distance
    approaches the cutoff and is zero outside the cutoff.

    Args:
        r: Distance values.
        cutoff: Maximum interaction distance.

    Returns:
        Cutoff weights with the same shape as ``r``.
    """
    return 0.5 * (torch.cos(torch.pi * r / cutoff) + 1.0) * (r < cutoff).float()


def mlp(in_dim, 
        hidden_dim, 
        out_dim,
        act_function='relu'):

    """Construct a **three-layer multilayer perceptron**.

    The network consists of two hidden linear layers with activation
    functions and a final linear output layer.

    Args:
        in_dim: Number of input features.
        hidden_dim: Number of hidden features.
        out_dim: Number of output features.
        act_function: Activation function used in hidden layers.

    Returns:
        A PyTorch Sequential module representing the MLP.
    """
    
    return Sequential(
        Linear(in_dim, hidden_dim),
        make_act_function(act_function)(),
        Linear(hidden_dim, hidden_dim),
        make_act_function(act_function)(),
        Linear(hidden_dim, out_dim)
        )


class FFNN(Module):
    """**Single-layer feed-forward neural network** with activation."""

    def __init__(
            self,
            input_dim,
            output_dim,
            act_func='relu'
    ):

        """Initialize the feed-forward network.
            Args:
                input_dim: Number of input features.
                output_dim: Number of output features.
                act_func: Activation function applied after the linear layer.
        """
        super().__init__()

        self.linear = Linear(input_dim, output_dim)
        self.act_func = make_act_function(
            act_func
        )()


    def forward(
            self,
            x
    ):
        x = self.linear(x)
        x = self.act_func(x)

        return x


class MLP_MODULE(Module):
    """**Multilayer perceptron** with configurable hidden layers, dropout, and activation function."""

    def __init__(
        self,
        in_dim,
        hidden_dim,
        output_dim,
        n_hidden_layers=2,
        act_function='relu',
        dropout: float=0.1
    ):
        """Initialize the multilayer perceptron.

        Args:
            in_dim: Number of input features.
            hidden_dim: Number of features in each hidden layer.
            output_dim: Number of output features.
            n_hidden_layers: Number of hidden feed-forward layers.
            act_function: Activation function used in hidden layers.
            dropout: Dropout probability applied after each hidden layer.
        """
        
        super().__init__()

        self.layers = ModuleList()

        self.layers.append(
            FFNN(in_dim, hidden_dim, act_func=act_function)
        )

        for _ in range(n_hidden_layers - 1):
            self.layers.append(
                FFNN(hidden_dim, hidden_dim, act_func=act_function)
            )

        self.output_layer = Linear(hidden_dim, output_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)

        return self.output_layer(x)



class DropoutEdge(Module):
    """Apply dropout to graph edges during training. Inspired by original work [DropEdge](https://arxiv.org/abs/1907.10903)"""

    def __init__(self, dropout_p, undirected=True):

        """Initialize edge dropout.

        Args:
            dropout_p: Probability of dropping an edge.
            undirected: If True, preserve undirected graph connectivity
                by dropping both directions of an edge together.
        """

        super().__init__()

        self.dropout_p = dropout_p
        self.undirected = undirected

    def forward(self, edge_index):
        return dropout_edge(edge_index, self.dropout_p, force_undirected=self.undirected, training=self.training)



class RBF(Module):

    """Expand scalar distances using Gaussian radial basis functions."""

    def __init__(self, n_rbf,cutoff=5):

        """Initialize Gaussian radial basis functions.

        Args:
            n_rbf: Number of radial basis functions.
            cutoff: Maximum distance used to distribute basis centers.
        """

        super().__init__()

        self.n_rbf = n_rbf
        centers = torch.linspace(0, cutoff, n_rbf, dtype=torch.float32)
        self.gamma = cutoff
        self.register_buffer('centers', centers)

    def forward(self, norm):

        rbf = torch.exp(-self.gamma * (norm - self.centers) ** 2)
        return rbf