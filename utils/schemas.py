from dataclasses import dataclass, asdict, fields, field
from collections.abc import Callable

from typing import Any, Sequence

from pathlib import Path

from rdkit import Chem
import numpy as np

from utils import general




def option_checker(obj):
    for field in fields(obj):
        if 'options' in field.metadata:
            options = field.metadata['options']
            option_name = field.name

            value = getattr(obj, option_name)

            if value not in options:
                raise ValueError(f'Unrecognized option <{value}> in <{option_name}> option')



def make_option_field(default_value: Any,
                      options: Sequence[Any]):
    
    return field(default=default_value,
                 metadata={'options': set(options)}) 



def _object_encoder(obj):

    """
    Converts Path objects into str
    
    """

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {k: _object_encoder(v) for k, v in obj.items()}

    return obj



@dataclass
class MolPaths:
    ligand_path: Path = None
    protein_path: Path = None


@dataclass
class Data:
    mol: Chem.Mol | None=None
    atom_features: np.ndarray | None=None
    bond_features: np.ndarray | None=None
    edge_index: np.ndarray | None=None
    positions: np.ndarray | None=None




@dataclass
class LigandData(Data):
    pass


@dataclass
class ProteinData(Data):
    pass



@dataclass
class Features:
    ligand_features: dict[str, Callable] = None
    protein_features: dict[str, Callable] = None



@dataclass
class DataSetSplited:
    train: list[Data] = None
    val: list[Data] = None
    test: list[Data] = None


@dataclass
class MetaData:
    def save(self, path: Path):
        
        general.save_json(_object_encoder(asdict(self)),
                          path)
        
