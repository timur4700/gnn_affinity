from dataclasses import dataclass
from schemas.general import MetaData

from pathlib import Path
from utils.options import make_path_field

from typing import Any


@dataclass
class ModelMetaData(MetaData):

    metadata_name: str = 'Model Metadata'

    model_name: str = ''
    id: str = None
    model_params: dict[str, Any] = None
    model_saved_params: Path = make_path_field("Directory with saved model's parameters")
    dataset_metadata: dict = None
    model_config_path: Path = make_path_field('Model Configuration File')
    trainer_config_path: Path = make_path_field('Training Configuration File')