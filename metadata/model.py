from dataclasses import dataclass
from metadata.base import MetaData

from pathlib import Path
from utils.options import make_path_field

from typing import Any, Annotated, ClassVar

from pydantic import Field

from metadata.graphs import GraphDatasetMeta


class ModelMetaData(MetaData):

    metadata_name: str = 'Model Metadata'
    _metadata_name: ClassVar[str] = 'Model Metadata'

    model_name: str | None = None
    id: str | None = None
    model_params: dict[str, Any] | None = None
    model_saved_params: Annotated[Path | None, Field(description="Directory with saved model's parameters")] = None
    dataset_metadata: GraphDatasetMeta | None = None
    model_config_path: Annotated[Path | None, Field(description='Model Configuration File')] = None
    trainer_config_path: Annotated[Path | None, Field(description='Training Configuration File')] = None