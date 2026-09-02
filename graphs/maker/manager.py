from typing import Any
from datasets import path_builders
from utils import schemas


from graphs.maker.builders import GraphBuilder

from pathlib import Path

from rdkit import Chem
from chem import protein_chem, utils

from datasets.pdbbind.metadata import MolConfig

from dataclasses import asdict




class GraphManager():
    def __init__(self,
                 path_builder: path_builders.PathBuilder,
                 graph_builder: GraphBuilder,
                 target_data: dict[str, Any],
                 mol_config: MolConfig,
                 sanitize: bool=False):


        # Defining Path 2 Files
        self.path_builder = path_builder

        # Graph Builder
        self.graph_builder = graph_builder

        self.mol_config = mol_config

        self.index_data = target_data
        self.sanitize = sanitize


    def set_data(self, complex_dir: str | Path):

        self.complex_dir = complex_dir if isinstance(complex_dir, Path) else Path(complex_dir)
        self.mol_paths = self.path_builder.build(self.complex_dir)

        # Complex ID, stated by complex directory name
        self.complex_name = self.complex_dir.name
        
    def load_protein(self,
                     ligand: Chem.Mol=None,
                     extract_pocket: bool=True,
                     extract_method: str='atom',
                     pocket_cutoff: float=10.0,
                     sanitize: bool=False) -> Chem.Mol:

        
        protein = Chem.MolFromPDBFile(self.mol_paths.protein_path, 
                                      sanitize=sanitize)

        if extract_pocket:
            return protein_chem.pocket_extraction(ligand,
                                                protein,
                                                cutoff_distance=pocket_cutoff,
                                                method=extract_method,
                                                sanitize=self.sanitize)


        return protein



    def load_ligand(self) -> Chem.Mol:

        """
        Ligand molecule loader function

        Returns
        -------
        rdkit.Chem.Mol
        """

        # Loading SDF, if ligand in .sdf format
        if  self.mol_paths.ligand_path.suffix == '.sdf':
            return utils.load_sdf(self.ligand_sdf, 
                                 sanitize=self.sanitize)[0]

        # Loading MOL2 if ligand in .mol2 format
        return utils.load_mol(self.mol_paths.ligand_path,
                             sanitize=self.sanitize)

    def extract_graph_data(self, complex_id):
        return self.index_data[complex_id]


    def init_graph_preparation(self):

        self.ligand_data = schemas.LigandData(mol=self.load_ligand())
        self.protein_data = schemas.ProteinData(mol=self.load_protein(self.ligand_data.mol,
                                                                      **asdict(self.mol_config.protein)))




        data = self.extract_graph_data(self.complex_name)

        graph = self.graph_builder.prepare_graph(self.ligand_data,
                                                 self.protein_data,
                                                 data=data)

        return graph
