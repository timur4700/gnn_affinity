from dataclasses import dataclass

from typing import Any, Self

from schemas.general import MetaData
from schemas.mol import Features

import os
from pathlib import Path

import pandas as pd

from pydantic import BaseModel, ConfigDict

from configs.mol import MolConfig
from configs.graph import ComplexGraphConfig



class SavingPaths(BaseModel):
    tmp_dir: Path | None = None
    tmp_dataset_np: Path | None = None
    graph_dataset_dir: Path | None = None
    graph_dataset: Path | None = None
    metadata: Path | None = None
    error_log: Path | None = None

    def make_dir(self):
        os.makedirs(self.graph_dataset_dir)


    @classmethod
    def make(
        cls,
        destination_path: Path,
        dataset_id: str
    ) -> Self:

        save_paths = cls(
            graph_dataset_dir=destination_path / 
            f'prepared_data_{dataset_id}'
        )
        
        save_paths.graph_dataset = (
            save_paths.graph_dataset_dir / 
            f'pdbbind_graph_dataset_{dataset_id}.pkl'
        )
        
        save_paths.metadata = (
            save_paths.graph_dataset_dir / 
            f'metadata_{dataset_id}.json'
        )
        
        save_paths.error_log = (
            save_paths.graph_dataset_dir / 
            f'error_{dataset_id}.log'
        )
        
        return save_paths


class ConfigData(BaseModel):
    graph_config: ComplexGraphConfig | None = None
    mol_config: MolConfig | None = None
    features: Features | None = None   


class EntriedData(BaseModel):
    paths: list[Path] | None = None
    target: dict | None = None


class MpConfig(BaseModel):
    n_cpu: int = 0
    chunk_size: int = 0


class PreprocessingData(BaseModel):
    id: str | None = None
    model: Any | None = None
    saving_paths: SavingPaths | None = None
    configs: ConfigData | None = None
    entries: EntriedData | None = None
    mp_config: MpConfig | None = None