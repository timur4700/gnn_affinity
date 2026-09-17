from dataclasses import dataclass, field
from schemas.general import MetaData

from typing import Literal, Any, Annotated
from pathlib import Path

from utils.options import make_path_field

from pydantic import BaseModel, Field
from metadata.base import MetaData

from configs.graph import ComplexGraphConfig


class GraphMetadata(MetaData):

    metadata_name = 'Graph Configuration Metadata'

    node_dim: int=0
    edge_dim: int=0

    ligand_features: Annotated[list[str], Field(default_factory=list)]
    protein_features: Annotated[list[str], Field(default_factory=list)]

    graph_config: ComplexGraphConfig | None = None



class GraphDatasetMeta(MetaData):

    metadata_name = 'Graph Dataset Metadata'

    name: str | None=None
    id: str | None=None
    model: str | None=None
    status: Literal['writing', 'failed', 'completed']='writing'
    graph_num: int=0
    graph_config_metadata: Annotated[GraphMetadata | None, Field(default_factory=GraphMetadata)]
    dataset_path: Annotated[Path | None, Field(description='Prepared Graph Dataset')]

    @classmethod
    def load(cls, metadata_path):
        metadata = super().load_from_file(metadata_path)
        print(f'Model: {metadata.model}')

        return metadata