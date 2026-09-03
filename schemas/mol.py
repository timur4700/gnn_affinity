from dataclasses import dataclass
import numpy as np

from pathlib import Path
from typing import Any
from collections.abc import Callable

from rdkit import Chem


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
class MolGraph:
    x: np.ndarray = None
    mol_id: np.ndarray = None
    pos: np.ndarray = None
    edge_index: np.ndarray = None
    edge_type: np.ndarray = None
    data: dict[str, Any] = None