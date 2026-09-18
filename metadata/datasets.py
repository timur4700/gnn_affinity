from typing import Annotated, ClassVar
from metadata.base import MetaData
from pydantic import Field

from pathlib import Path



class DatasetMetadata(MetaData):

    metadata_name: str = 'Dataset'
    _metadata_name: ClassVar[str] = 'Dataset'

    name: str = ''
    entries_path: Annotated[
        Path | None, 
        Field(
        default=None,
        description='Entries Directory', 
        json_schema_extra={'file_required': True}
        )
    ] = None
    
    target_path: Annotated[
        Path | None, 
        Field(
            default=None,
            description='Target Data File',
            json_schema_extra={'file_required': True}
        )
    ] = None

    model_name: str = ''
    graph_name: str = ''

    graph_config_path: Annotated[
        Path | None, 
        Field(
            default=None,
            description='Graph Configuration File',
            json_schema_extra={'file_required': True}
        )
    ] = None
    
    mol_config_path: Annotated[
        Path | None, 
        Field(
            description='Molecule Configuration File',
            json_schema_extra={'file_required': True}
        )
    ] = None