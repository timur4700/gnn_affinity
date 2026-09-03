from dataclasses import dataclass
from utils.schemas import MetaData



@dataclass
class DatasetMetadata(MetaData):

    name: str = ''
    entries_path: str = ''
    target_path: str = ''
    model: str = ''
    graph: str = ''
    graph_config_path: str = ''
    mol_config_path: str = ''

