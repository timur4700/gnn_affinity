from dataclasses import dataclass, field, asdict
from typing import Literal

from pathlib import Path

from utils import schemas, graphs, general

@dataclass
class GraphConfig:
    node_dim: int=0
    edge_dim: int=0

    ligand_features: list[str]=field(default_factory=list)
    protein_features: list[str]=field(default_factory=list)

    graph_type: Literal['2d', '3d']='2d'
    include_edge_attr: bool=False
    include_inter_edges: bool=False
    intra_cutoff: float=0.0
    inter_cutoff: float=0.0




@dataclass
class PDBdatasetMeta:
    tag: str='pdbbind_dataset'
    id: str=''
    model: str=''
    status: Literal['writing', 'failed', 'completed']='writing'
    graph_num: int=0
    graph_config: GraphConfig=field(default_factory=GraphConfig)
    dataset_path: Path=None

    def save(self, path):
        self.dataset_path = str(self.dataset_path)
        general.save_json(asdict(self),
                          path)



def make_graph_config(features: schemas.Features,
                      global_config: dict[str | float | bool]) -> GraphConfig:

    graph_global_config = global_config['graph_preparation']

    node_dim = min(len(features.ligand_features),
                   len(features.protein_features)) + graphs.add_feature_padding(features.ligand_features,
                                                                                features.protein_features)

    

    include_inter_edges = graph_global_config['comb_graph']['add_interaction_edges']
    inter_cutoff = graph_global_config['comb_graph']['cutoff']


    return GraphConfig(node_dim=node_dim,
                       ligand_features=list(features.ligand_features.keys()),
                       protein_features=list(features.protein_features.keys()),
                       include_inter_edges=include_inter_edges,
                       inter_cutoff=inter_cutoff)



