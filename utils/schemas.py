from dataclasses import dataclass, asdict
from collections.abc import Callable

from pathlib import Path

from rdkit import Chem
import numpy as np

from utils import general




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
class PLAData:
    ligand: LigandData | None=None
    protein: ProteinData | None=None
    interaction_edges: np.ndarray | None=None
    target: int | float | np.ndarray | None=None


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
        
