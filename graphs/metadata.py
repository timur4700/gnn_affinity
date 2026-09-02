from dataclasses import dataclass, field
from utils.schemas import MetaData, Features

from typing import Literal, Any
from pathlib import Path

from graphs.utils import add_feature_padding
from graphs.configs.structure import GraphConfig

from utils.schemas import Features

import pandas as pd

import os

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



@dataclass
class GraphDatasetMeta(MetaData):
    tag: str=''
    id: str=''
    model: str=''
    status: Literal['writing', 'failed', 'completed']='writing'
    graph_num: int=0
    graph_config_metadata: GraphMetadata=field(default_factory=GraphMetadata)
    dataset_path: Path=None


@dataclass
class SavingPaths:
    tmp_dir: str = ''
    tmp_dataset_np: str = ''
    graph_dataset_dir: str = ''
    graph_dataset: str = ''
    metadata: str = ''
    error_log: str = ''

    @classmethod
    def make(cls, destination_path, dataset_id):

        save_paths = cls(graph_dataset_dir=destination_path / \
                         f'prepared_data_{dataset_id}')


        save_paths.graph_dataset = save_paths.graph_dataset_dir / \
                                    f'pdbbind_graph_dataset_{dataset_id}.pkl'

        save_paths.metadata = save_paths.graph_dataset_dir / \
                                f'metadata_{dataset_id}.json'

        save_paths.error_log = save_paths.graph_dataset_dir / \
                                f'error_{dataset_id}.log'

        return save_paths


    def make_dir(self):
        os.makedirs(self.graph_dataset_dir)


@dataclass
class ConfigData:
    graph_config: Any = None
    mol_config: Any = None
    features: Features = None



@dataclass
class EntriedData:
    paths: list[str] = None
    target: pd.DataFrame = None



@dataclass
class MpConfig:
    n_proc: int = 0
    chunk_size: int = 0



@dataclass
class PreprocessingData(MetaData):
    id: str = ''
    saving_paths: SavingPaths = None
    configs: ConfigData = None
    entries: EntriedData = None
    mp_config: MpConfig = None

@dataclass
class PostProcessingData(MetaData):
    pass




def make_graph_metadata(features: Features,
                        graph_config: GraphConfig) -> GraphMetadata:


    node_dim = min(len(features.ligand_features),
                   len(features.protein_features)) + add_feature_padding(features.ligand_features,
                                                                        features.protein_features)

    

    include_inter_edges = graph_config.interaction['add_interaction_edges']
    inter_cutoff = graph_config.interaction['cutoff']


    return GraphMetadata(node_dim=node_dim,
                         ligand_features=list(features.ligand_features.keys()),
                         protein_features=list(features.protein_features.keys()),
                         include_inter_edges=include_inter_edges,
                         inter_cutoff=inter_cutoff)



