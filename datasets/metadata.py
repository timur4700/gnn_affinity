from dataclasses import dataclass
from utils.schemas import MetaData



@dataclass
class DatasetMeta(MetaData):
    name: str = ''
    entries_path: str = ''
    target_path: str = ''
    