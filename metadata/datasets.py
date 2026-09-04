from dataclasses import dataclass
from schemas.general import MetaData

from pathlib import Path
from utils.options import make_path_field

@dataclass
class DatasetMetadata(MetaData):

    metadata_name: str = 'Dataset Metadata'

    name: str = ''
    entries_path: Path = make_path_field('Entries Directory')
    target_path: Path = make_path_field('Target Data File')
    model: str = ''
    graph: str = ''
    graph_config_path: Path = make_path_field('Graph Configuration File')
    mol_config_path: Path = make_path_field('Molecule Configuration File')