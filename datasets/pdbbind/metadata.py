from dataclasses import dataclass
from typing import Literal



@dataclass
class LigandConfig:
    ligand_format: Literal['sdf, mol2'] = ''



@dataclass
class ProteinConfig:
    type: str = ''
    cutoff: float = 0
    method: Literal['atom', 'cog', 'com'] = ''



@dataclass
class MolConfig:

    ligand: LigandConfig
    protein: ProteinConfig

    @classmethod
    def load_data(cls, data: dict):
        return cls(
            ligand=LigandConfig(**data['Ligand']),
            protein=ProteinConfig(**data['Protein'])
        )


