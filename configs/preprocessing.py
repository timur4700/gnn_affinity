from dataclasses import dataclass

from typing import Any

from schemas.general import MetaData
from schemas.mol import Features

import os
from pathlib import Path

import pandas as pd




@dataclass
class SavingPaths:
    tmp_dir: Path = ''
    tmp_dataset_np: Path = ''
    graph_dataset_dir: Path = ''
    graph_dataset: Path = ''
    metadata: Path = ''
    error_log: Path = ''

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
    n_cpu: int = 0
    chunk_size: int = 0



@dataclass
class PreprocessingData(MetaData):
    id: str = ''
    model: Any = None
    saving_paths: SavingPaths = None
    configs: ConfigData = None
    entries: EntriedData = None
    mp_config: MpConfig = None

    

@dataclass
class PostProcessingData(MetaData):
    pass
