from dataclasses import dataclass
from schemas.general import MetaData
from metadata.base import MetaDataPy

from pathlib import Path
from utils.options import make_path_field

from pydantic import BaseModel, Field
from typing import Annotated


class DatasetMetadata(MetaData):

    metadata_name: str = 'Dataset Metadata'

    name: str = ''
    entries_path: Path = make_path_field('Entries Directory')
    target_path: Path = make_path_field('Target Data File')
    model: str = ''
    graph: str = ''
    graph_config_path: Path = make_path_field('Graph Configuration File')
    mol_config_path: Path = make_path_field('Molecule Configuration File')



class DatasetMetadataPy(MetaDataPy):

    metadata_name: str = 'Dataset Metadata'
    name: str = ''
    entries_path: Annotated[Path, Field(description='Entries Directory')]
    target_path: Annotated[Path, Field(description='Target Data File')]
    model_name: str = ''
    graph: str = ''
    graph_config_path: Annotated[Path, Field(description='Graph Configuration File')]
    mol_config_path: Annotated[Path, Field(description='Molecule Configuration File')]