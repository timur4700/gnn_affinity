from rdkit import Chem

import numpy as np

import torch
from torch_geometric.data import Data

from utils import schemas, chem

from collections.abc import Callable
from typing import Any



def add_feature_padding(ligand_features: dict[str, Callable],
                         protein_features: dict[str, Callable]):


    if len(ligand_features) == len(protein_features):
        return 0

    return int(abs(len(ligand_features) - len(protein_features)))



def add_ligand_protein_id(ligand_x: torch.Tensor,
                        protein_x: torch.Tensor):

    ligand_id = 0
    protein_id = 1

    ligand_id = torch.tensor([ligand_id] * ligand_x.shape[0],
                             dtype=torch.long).reshape(-1, 1)

    protein_id = torch.tensor([protein_id] * protein_x.shape[0],
                              dtype=torch.long).reshape(-1, 1)

    return torch.cat([ligand_id, protein_id], dim=0)



def construct_feature_matrix(
        mol: Chem.Mol,
        ligand=False,
        ligand_features: dict[str, Callable]=None,
        protein_features: dict[str, Callable]= None,
        protein: bool = False) -> tuple[np.ndarray, np.ndarray]:
    

    """
    Constructs feature matrix per protein [N, F]

    Current features and associated column ID

    - 0: Atomic Number, int
    - 1: Aromatic Type [0, 1], int
    - 2: Number of Total Attached Hydrogens (explicit + implicit), int
    - 3: Residue Type (for protein), int

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object

    schema : dataclass
        The dataclass for Ligand or Protein Data

    Returns
    -------
    np.ndarray
        Constructed feature matrix
    """

    n_atoms = mol.GetNumAtoms()

    ligand_features = ligand_features or chem.AtomFeatureExtract().extract_func()
    protein_features = protein_features or chem.ProteinFeatureExtract().extract_func()

    feature_dict = ligand_features if ligand else protein_features

    matrix = np.zeros((n_atoms,
                       min(len(ligand_features), len(protein_features))
                       + add_feature_padding(ligand_features,
                                             protein_features)))


    # Extraction of Atomic Positions (3D)
    positions = mol.GetConformer().GetPositions()

    for i, atom in enumerate(mol.GetAtoms()):
         for n,f in enumerate(feature_dict.values()):
              matrix[i, n] = f(atom)

    return (matrix, positions)



def construct_edge_matrix(mol: Chem.Mol,
                          **kwargs) -> schemas.LigandData | schemas.ProteinData:

    a_matrix = chem.adjacency_matrix(mol, **kwargs)
    edge_index = chem.to_sparse_adjacency_matrix(a_matrix)


    return edge_index



def construct_interaction_edges(ligand_positions: np.ndarray | torch.Tensor,
                                protein_positions: np.ndarray | torch.Tensor,
                                cutoff: int | float=5) -> np.ndarray | torch.Tensor:


    d_ij = chem.get_distance(ligand_positions,
                             protein_positions)

    cutoff_mask = d_ij <= cutoff

    if (isinstance(ligand_positions, torch.Tensor) and
        isinstance(protein_positions, torch.Tensor)):

        return torch.nonzero(cutoff_mask).T

    return np.array(np.nonzero(cutoff_mask))




def combine_edge_index(
    ligand_edge_index: torch.Tensor,
    protein_edge_index: torch.Tensor,
    num_ligand_nodes: int,
    interaction_edges: torch.Tensor=None,
) -> torch.Tensor:

    protein_edge_index = protein_edge_index + num_ligand_nodes


    if interaction_edges is not None:

        interaction_edges = interaction_edges.clone()
        interaction_edges[1] += num_ligand_nodes

        return torch.cat(
            [
                ligand_edge_index,
                protein_edge_index,
                interaction_edges,
            ],
            dim=1,
        )


    return torch.cat([
        ligand_edge_index,
        protein_edge_index
    ], dim=1)



def add_edge_type(*args):

    index_tensor = []

    for i, tensor in enumerate(args):

        if tensor is None:
            continue

        index_tensor.append(torch.tensor(
            [i] * tensor.shape[-1], dtype=torch.float32
        ))


    return torch.cat(index_tensor).reshape(-1, 1)


def add_edge_dist(edge_index: torch.Tensor,
                  positions):

    i, j = edge_index
    pos_i = positions[i]
    pos_j = positions[j]

    d_ij = chem.get_distance(pos_i, pos_j)


    return d_ij



def prepare_mol_data(data: schemas.LigandData | schemas.ProteinData,
                     undirected=True,
                     self_loop=False,
                     features: schemas.Features = None):

    """
    Main data preparation function
    
    """

    if isinstance(data, schemas.LigandData):
        ligand = True

    elif isinstance(data, schemas.ProteinData):
        ligand = False

    else:
        raise ValueError('Invalid schema')


    # Preparing Atom (Node) features and Positions
    data.atom_features, data.positions = construct_feature_matrix(
                                            data.mol,
                                            ligand,
                                            features.ligand_features,
                                            features.protein_features)


    # Preparing Edge Index
    data.edge_index = construct_edge_matrix(data.mol,
                                            undirected=undirected,
                                            self_loop=self_loop)



    return data



def make_graph(data: schemas.LigandData | schemas.ProteinData,
                  undirected=True,
                  self_loop=False,
                  features: schemas.Features = None,
                  cutoff: float=None) -> Data:

    graph = Data()

    features = features if features is not None else schemas.Features()

    data = prepare_mol_data(data, 
                            undirected=undirected,
                            self_loop=self_loop,
                            features=features)

    data.convert2tensor()


    graph.x = data.atom_features
    graph.pos = data.positions
    graph.edge_index = data.edge_index


    return graph

    

def combine_graphs(ligand_graph: Data,
                   protein_graph: Data,
                   add_interaction_edges: bool=False,
                   edge_type: bool=False,
                   cutoff: int | float=5) -> Data:

    graph= Data()
    interaction_edges = None


    graph.x = torch.cat([ligand_graph.x,
                   protein_graph.x],
                   dim=0)



    graph.pos = torch.cat([ligand_graph.pos,
                     protein_graph.pos],
                     dim=0)

    graph.mol_id = add_ligand_protein_id(
        ligand_graph.x,
        protein_graph.x
    )


    if add_interaction_edges:
        interaction_edges = construct_interaction_edges(ligand_graph.pos,
                                                        protein_graph.pos,
                                                        cutoff=cutoff)

        if isinstance(interaction_edges, np.ndarray):
            interaction_edges = torch.tensor(interaction_edges, 
                                             dtype=torch.long)


    graph.edge_index = combine_edge_index(ligand_graph.edge_index,
                                    protein_graph.edge_index,
                                    ligand_graph.x.shape[0],
                                    interaction_edges)


    # 0 -> ligand bonds; 1 -> protein bonds -> 2 interaction between ligand and protein

    if edge_type:
        graph.edge_type = add_edge_type(ligand_graph.edge_index,
                              protein_graph.edge_index,
                              interaction_edges)


    return graph



def insert_data2graph(graph: Data,
                      data: dict[str, Any],
                      y_dtype: torch.dtype=torch.float32):

    """
    The function that unserts data to torch_geometric.data.Data object
    **Note!** The function expect, that data includes ``y`` key, ie target variable
    
    """

    if 'y' not in data:
        print('The target variable was not provided for the graph')

    for k, v in data.items():

        if k == 'y':
            setattr(graph, k, torch.tensor(v, dtype=y_dtype))
            continue

        setattr(graph, k, v)