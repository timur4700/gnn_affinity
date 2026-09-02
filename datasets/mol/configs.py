from dataclasses import dataclass, field
from typing import Literal
from utils import schemas





@dataclass
class Formats:
    ligand: Literal['sdf', 'mol2'] = schemas.make_option_field('mol2',
                                                               ['mol2', 'sdf'])

    def __post_init__(self):
        schemas.option_checker(self)


@dataclass
class LigandConfig:
    sanitize: bool = schemas.make_option_field(False,
                                               [True, False])

@dataclass
class ProteinConfig:
    extract_pocket: bool = schemas.make_option_field(True,
                                                     [True, False])


    extract_method: Literal['atom', 'cog', 'com'] = schemas.make_option_field('atom',
                                                                              ['atom', 'cog', 'com'])
    pocket_cutoff: float=10.0
    sanitize: bool = False

    def __post_init__(self):
        schemas.option_checker(self)



@dataclass
class MolConfig:

    formats: Formats
    ligand: LigandConfig
    protein: ProteinConfig

    @classmethod
    def load_data(cls,
                  data: dict
):
        return cls(
            formats=Formats(**data['Formats']),
            ligand=LigandConfig(**data['Ligand']),
            protein=ProteinConfig(**data['Protein'])
        )


@dataclass
class PDBbindMolConfig(MolConfig):
    prot_source: Literal['pocket', 'protein'] = schemas.make_option_field('pocket',
                                                                          ['pocket', 'protein'])


    def __post_init__(self):
        if self.prot_source not in {'pocket', 'protein'}:
            raise ValueError(f'Unrecognized Option: {self.prot_source}')


    @classmethod
    def load_data(cls, data):
        mol_base = super().load_data(data)

        return cls(
            formats=mol_base.formats,
            ligand=mol_base.ligand,
            protein=mol_base.protein,
            prot_source=data['ProteinSource']
        )
    