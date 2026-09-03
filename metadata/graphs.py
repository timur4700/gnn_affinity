from dataclasses import dataclass, field
from schemas.general import MetaData

from typing import Literal, Any
from pathlib import Path


@dataclass
class GraphMetadata(MetaData):
    node_dim: int=0
    edge_dim: int=0

    ligand_features: list[str]=field(default_factory=list)
    protein_features: list[str]=field(default_factory=list)

    graph_config: Any = None



@dataclass
class GraphDatasetMeta(MetaData):
    name: str=''
    id: str=''
    model: str=''
    status: Literal['writing', 'failed', 'completed']='writing'
    graph_num: int=0
    graph_config_metadata: GraphMetadata=field(default_factory=GraphMetadata)
    dataset_path: Path=None