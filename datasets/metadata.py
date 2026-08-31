from dataclasses import dataclass
from utils.schemas import MetaData



@dataclass
class DatasetMetadata(MetaData):

    # Dataset Name
    name: str = ''

    # Directory with PL-complexes entries
    entries_path: str = ''

    # Path, where target variable stores (.csv)
    target_path: str = ''

    # DL model name
    model: str = ''

    # Graph Name
    graph: str = ''

    # Path to graph configuration file
    graph_config_path: str = ''

    # Path to molecule configuration file
    mol_config_path: str = ''