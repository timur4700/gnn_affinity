from dataclasses import dataclass, field
from utils.schemas import MetaData

from typing import Literal



@dataclass
class GraphMetadata(MetaData):
    node_dim: int=0
    edge_dim: int=0

    ligand_features: list[str]=field(default_factory=list)
    protein_features: list[str]=field(default_factory=list)

    graph_type: Literal['2d', '3d']='2d'
    include_edge_attr: bool=False
    include_inter_edges: bool=False
    intra_cutoff: float=0.0
    inter_cutoff: float=0.0