from dataclasses import dataclass



@dataclass
class ModelSettings:
    n_rbf: int = 32
    hidden_dim: int = 64
    output_dim: int = 1
    cutoff: float = 5.0
    dropout: float = 0.1
    n_gine: int = 2
    n_egnn: int = 2
    normalization: str = 'batchnorm'
