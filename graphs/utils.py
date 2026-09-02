import numpy as np
from collections.abc import Callable
from chem.schemas import MolGraph
from chem import utils as chem_utils
from rdkit import Chem

from utils import schemas

from typing import Any




def add_feature_padding(ligand_features: dict[str, Callable],
                         protein_features: dict[str, Callable]):


    if len(ligand_features) == len(protein_features):
        return 0

    return int(abs(len(ligand_features) - len(protein_features)))



def add_ligand_protein_id(ligand_x: np.ndarray,
                        protein_x: np.ndarray):

    ligand_id = 0
    protein_id = 1

    ligand_id = np.array([ligand_id] * ligand_x.shape[0],
                         dtype=np.int64).reshape(-1, 1)

    protein_id = np.array([protein_id] * protein_x.shape[0],
                              dtype=np.int64).reshape(-1, 1)

    return np.concatenate([ligand_id, protein_id], axis=0)



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

    ligand_features = ligand_features or chem_utils.AtomFeatureExtract().extract_func()
    protein_features = protein_features or chem_utils.ProteinFeatureExtract().extract_func()

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

    a_matrix = chem_utils.adjacency_matrix(mol, **kwargs)
    edge_index = chem_utils.to_sparse_adjacency_matrix(a_matrix)


    return edge_index



def construct_interaction_edges(ligand_positions: np.ndarray,
                                protein_positions: np.ndarray,
                                cutoff: int | float=5) -> np.ndarray:


    d_ij = chem_utils.get_distance(ligand_positions,
                             protein_positions)

    cutoff_mask = d_ij <= cutoff

    return np.array(np.nonzero(cutoff_mask))




def combine_edge_index(
    ligand_edge_index: np.ndarray,
    protein_edge_index: np.ndarray,
    num_ligand_nodes: int,
    interaction_edges: np.ndarray=None,
) -> np.ndarray:

    protein_edge_index = protein_edge_index + num_ligand_nodes


    if interaction_edges is not None:

        interaction_edges = interaction_edges.copy()
        interaction_edges[1] += num_ligand_nodes

        return np.concatenate(
            [
                ligand_edge_index,
                protein_edge_index,
                interaction_edges,
            ],
            axis=1,
        )


    return np.concatenate([
        ligand_edge_index,
        protein_edge_index
    ], axis=1)



def add_edge_type(*args):

    index_array = []

    for i, array in enumerate(args):

        if array is None:
            continue

        index_array.append(np.array(
            [i] * array.shape[-1], dtype=np.float32))


    return np.concatenate(index_array).reshape(-1, 1)


def add_edge_dist(edge_index: np.ndarray,
                  positions: np.ndarray):

    i, j = edge_index
    pos_i = positions[i]
    pos_j = positions[j]

    d_ij = chem_utils.get_distance(pos_i, pos_j)


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
                  cutoff: float=None) -> MolGraph:

    mol_graph = MolGraph()

    features = features if features is not None else schemas.Features()

    data = prepare_mol_data(data, 
                            undirected=undirected,
                            self_loop=self_loop,
                            features=features)


    mol_graph.x = data.atom_features
    mol_graph.pos = data.positions
    mol_graph.edge_index = data.edge_index


    return mol_graph

    

def combine_graphs(ligand_graph: MolGraph,
                   protein_graph: MolGraph,
                   add_interaction_edges: bool=False,
                   edge_type: bool=False,
                   cutoff: int | float=5) -> MolGraph:

    mol_graph = MolGraph()
    interaction_edges = None


    mol_graph.x = np.concatenate([ligand_graph.x,
                   protein_graph.x],
                   axis=0)



    mol_graph.pos = np.concatenate([ligand_graph.pos,
                     protein_graph.pos],
                     axis=0)

    mol_graph.mol_id = add_ligand_protein_id(
        ligand_graph.x,
        protein_graph.x
    )


    if add_interaction_edges:
        interaction_edges = construct_interaction_edges(ligand_graph.pos,
                                                        protein_graph.pos,
                                                        cutoff=cutoff)

        if isinstance(interaction_edges, np.ndarray):
            interaction_edges = np.array(interaction_edges, 
                                             dtype=np.int64)


    mol_graph.edge_index = combine_edge_index(ligand_graph.edge_index,
                                    protein_graph.edge_index,
                                    ligand_graph.x.shape[0],
                                    interaction_edges)


    # 0 -> ligand bonds; 1 -> protein bonds -> 2 interaction between ligand and protein

    if edge_type:
        mol_graph.edge_type = add_edge_type(ligand_graph.edge_index,
                              protein_graph.edge_index,
                              interaction_edges)


    return mol_graph



def insert_data2graph(graph: MolGraph,
                      data: dict[str, Any]):

    """
    The function that unserts data to torch_geometric.data.Data object
    **Note!** The function expect, that data includes ``y`` key, ie target variable
    
    """

    if 'y' not in data:
        print('The target variable was not provided for the graph')

    for k, v in data.items():

        if k == 'y':
            setattr(graph, k, v)
            continue

        setattr(graph, k, v)