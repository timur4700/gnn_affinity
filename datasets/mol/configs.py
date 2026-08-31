from dataclasses import dataclass
from typing import Literal




@dataclass
class LigandConfig:
    ligand_format: Literal['sdf, mol2'] = ''
    sanitize: bool = False


@dataclass
class ProteinConfig:
    cutoff: float = 0.0
    method: Literal['atom', 'cog', 'com'] = ''
    sanitize: bool = False



@dataclass
class MolConfig:

    ligand: LigandConfig
    protein: ProteinConfig

    @classmethod
    def load_data(cls,
                  data: dict
):
        return cls(
            ligand=LigandConfig(**data['Ligand']),
            protein=ProteinConfig(**data['Protein'])
        )



@dataclass
class PDBbindMolConfig(MolConfig):
    prot_source: str = ''

    @classmethod
    def load_data(cls, data):
        mol_base = super().load_data(data)

        return cls(
            ligand=mol_base.ligand,
            protein=mol_base.protein,
            prot_source=data['ProteinSource']
        )
    