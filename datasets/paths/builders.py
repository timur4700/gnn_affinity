from utils import schemas
from pathlib import Path
from typing import Literal
from abc import ABC, abstractmethod

from datasets.mol import configs



class PathBuilder(ABC):

    """
    The base method for building paths to the ligand and protein, based on 
    directory of PL-complex
    """

    def __init__(self):

        self.protein_format = None
        self.ligand_format = None


    @abstractmethod
    def build(self, complex_dir) -> schemas.MolPaths:
        pass



class PDBbindPaths(PathBuilder):
    def __init__(self,
                 pocket: bool=True,
                 ligand: 
                 Literal['mol2, sdf'] = 'mol2'):

        
        self.protein_format = '{}' + f"_{'pocket' if pocket else 'protein'}.pdb"
        self.ligand_format = '{}' + f'_ligand.{ligand}'

    @classmethod
    def make(cls, mol_config: configs.PDBbindMolConfig):

        is_pocket = mol_config.prot_source == 'pocket'

        if not is_pocket:
            if not mol_config.protein.extract_pocket:
                raise ValueError('Due to memory safety, turn on pocket extraction, when selecting <protein> as main source')

        return cls(
            pocket=is_pocket,
            ligand=mol_config.formats.ligand
        )


    def build(self, complex_dir: Path):
        pdb_name = complex_dir.name

        return schemas.MolPaths(ligand_path=complex_dir/self.ligand_format.format(pdb_name),
                                protein_path=complex_dir/self.protein_format.format(pdb_name))