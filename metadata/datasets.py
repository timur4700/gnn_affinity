from typing import Annotated, ClassVar
from metadata.base import MetaData
from pydantic import Field

from pathlib import Path



class DatasetMetadata(MetaData):

    metadata_name: str = 'Dataset'
    _metadata_name: ClassVar[str] = 'Dataset'

    name: str = ''
    entries_path: Annotated[Path | None, Field(description='Entries Directory')] = None
    target_path: Annotated[Path | None, Field(description='Target Data File')] = None
    model_name: str = ''
    graph_name: str = ''
    graph_config_path: Annotated[Path | None, Field(description='Graph Configuration File')] = None
    mol_config_path: Annotated[Path | None, Field(description='Molecule Configuration File')] = None