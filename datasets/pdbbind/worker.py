from rdkit import Chem

from pathlib import Path

import pandas as pd

from utils import protein_func, chem, schemas, graphs
import torch



class PDBWorker():
    def __init__(self, 
                 pdb_dir: str | Path,
                 index_data: pd.DataFrame,
                 config: dict[str, str | float],
                 features: schemas.Features = None,
                 sanitize: bool=False):


        # Defining Path 2 Files
        pdb_dir = pdb_dir if isinstance(pdb_dir, Path) else Path(pdb_dir)
        self.pdb_dir = pdb_dir
        self.pdb_name = pdb_dir.name

        self.ligand_sdf = pdb_dir / f"{self.pdb_name}_ligand.sdf"
        self.ligand_mol2 = pdb_dir / f"{self.pdb_name}_ligand.mol2"

        self.protein = pdb_dir / f"{self.pdb_name}_protein.pdb"
        self.pocket =  pdb_dir / f"{self.pdb_name}_pocket.pdb"

        self.config = config
        self.feautures = features

        self.index_data = index_data

        self.sanitize = sanitize

        
    def load_protein(self,
                     ligand: Chem.Mol, 
                     type: str,
                     cutoff: float,
                     method: str='atom') -> Chem.Mol:

        if type == 'pocket':
            return chem.load_mol(self.pocket, 
                                 sanitize=self.sanitize)


        protein = Chem.MolFromPDBFile(self.protein, 
                                      sanitize=self.sanitize)
        
        return protein_func.pocket_extraction(ligand,
                                              protein,
                                              cutoff_distance=cutoff,
                                              method=method,
                                              sanitize=self.sanitize)



    def load_ligand(self,
                    ligand_format) -> Chem.Mol:

        if ligand_format == 'sdf':
            return chem.load_sdf(self.ligand_sdf, 
                                 sanitize=self.sanitize)[0]

        return chem.load_mol(self.ligand_mol2,
                             sanitize=self.sanitize)

    def extract_index_data(self, pdb_name):
        return self.index_data[pdb_name]


    def init_graph_preparation(self):

        mol_prep_config = self.config['mol_preparation']

        self.ligand_data = schemas.LigandData(mol=self.load_ligand(**mol_prep_config['ligand']))

        self.protein_data = schemas.ProteinData(mol=self.load_protein(self.ligand_data.mol,
                                                                 **mol_prep_config
                                                                    ['protein']))

        graph_config = self.config['graph_preparation']

        ligand_graph = graphs.prepare_graph(self.ligand_data,
                                            **graph_config['single_graph'],
                                            features=self.feautures)
        
        protein_graph = graphs.prepare_graph(self.protein_data, 
                                             **graph_config['single_graph'],
                                             features=self.feautures)

        graph = graphs.combine_graphs(ligand_graph,
                                      protein_graph,
                                      **graph_config['comb_graph'])

        data = self.extract_index_data(self.pdb_name)

        graph.y = torch.tensor(data['y'], dtype=torch.float32)
        graph.dataset = data['dataset']
        graph.pdb_id = self.pdb_name

        return graph