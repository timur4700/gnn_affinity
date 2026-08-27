from dataclasses import dataclass
from collections.abc import Callable

import torch

from rdkit import Chem
import numpy as np

@dataclass
class Data:
    mol: Chem.Mol | None=None
    atom_features: np.ndarray | None=None
    bond_features: np.ndarray | None=None
    edge_index: np.ndarray | None=None
    positions: np.ndarray | None=None


    def convert2tensor(self):

        self.atom_features = torch.tensor(self.atom_features, 
                                          dtype=torch.float32)

        self.edge_index = torch.tensor(self.edge_index,
                                       dtype=torch.long)

        self.positions = torch.tensor(self.positions, 
                                      dtype=torch.float32)



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
