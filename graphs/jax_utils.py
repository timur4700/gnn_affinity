import jax.numpy as jnp
from jax import jit
from jax import random

from chem import jax_utils, mol_features
from graphs.utils import add_feature_padding


from rdkit import Chem

from collections.abc import Callable
from typing import Literal

from schemas import mol



def construct_feature_matrix_jax(
        mol: Chem.Mol,
        ligand=False,
        ligand_features: dict[str, Callable]=None,
        protein_features: dict[str, Callable]= None,
        protein: bool = False) -> tuple[jnp.ndarray, jnp.ndarray]:
    

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

    ligand_features = ligand_features or mol_features.AtomFeatureExtract().extract_func()
    protein_features = protein_features or mol_features.ProteinFeatureExtract().extract_func()

    feature_dict = ligand_features if ligand else protein_features

    matrix = jnp.zeros((n_atoms,
                       min(len(ligand_features), len(protein_features))
                       + add_feature_padding(ligand_features,
                                             protein_features)))


    # Extraction of Atomic Positions (3D)
    positions = mol.GetConformer().GetPositions()

    for i, atom in enumerate(mol.GetAtoms()):
         for n,f in enumerate(feature_dict.values()):
              matrix[i, n] = f(atom)

    return (matrix, positions)



def construct_edge_matrix_2d_jax(mol: Chem.Mol,
                          **kwargs) -> jnp.ndarray:

    a_matrix = jax_utils.adjacency_matrix_jax(mol, **kwargs)
    edge_index = jax_utils.to_sparse_adjacency_matrix_jax(a_matrix)

    return edge_index





@jit
def add_ligand_protein_id_jax(ligand_x: jnp.ndarray,
                              protein_x: jnp.ndarray):

    ligand_id = 0
    protein_id = 1

    ligand_id = jnp.array([ligand_id] * ligand_x.shape[0],
                         ).reshape(-1, 1)

    protein_id = jnp.array([protein_id] * protein_x.shape[0],
                              ).reshape(-1, 1)

    return jnp.concatenate([ligand_id, protein_id], axis=0)



@jit
def construct_edge_matrix_3d_jax(positions: jnp.ndarray,
                             intra_cutoff) -> jnp.ndarray:

    d_ij = jax_utils.get_distance_jax(positions,
                                   positions)

    cutoff_mask = d_ij <= intra_cutoff

    return jnp.array(jnp.nonzero(cutoff_mask))



@jit
def construct_interaction_edges_jax(ligand_positions: jnp.ndarray,
                                    protein_positions: jnp.ndarray,
                                    cutoff: int | float=5) -> jnp.ndarray:


    d_ij = jax_utils.get_distance_jax(ligand_positions,
                                   protein_positions)

    cutoff_mask = d_ij <= cutoff

    return jnp.array(jnp.nonzero(cutoff_mask))



@jit
def combine_edge_index_jax(
    ligand_edge_index: jnp.ndarray,
    protein_edge_index: jnp.ndarray,
    num_ligand_nodes: int,
    interaction_edges: jnp.ndarray=None,
) -> jnp.ndarray:

    protein_edge_index = protein_edge_index + num_ligand_nodes


    if interaction_edges is not None:

        interaction_edges = interaction_edges.copy()
        interaction_edges[1] += num_ligand_nodes

        return jnp.concatenate(
            [
                ligand_edge_index,
                protein_edge_index,
                interaction_edges,
            ],
            axis=1,
        )


    return jnp.concatenate([
        ligand_edge_index,
        protein_edge_index
    ], axis=1)



@jit
def add_edge_type_jax(*args):

    index_array = []

    for i, array in enumerate(args):

        if array is None:
            continue

        index_array.append(jnp.array(
            [i] * array.shape[-1], dtype=jnp.float32))


    return jnp.concatenate(index_array).reshape(-1, 1)



@jit
def add_edge_dist_jax(edge_index: jnp.ndarray,
                  positions: jnp.ndarray):

    i, j = edge_index
    pos_i = positions[i]
    pos_j = positions[j]

    d_ij = jax_utils.get_distance_jax(pos_i, pos_j)


    return d_ij





def prepare_mol_data(data: mol.LigandData | mol.ProteinData,
                     undirected=True,
                     self_loop=False,
                     features: mol.Features = None,
                     graph_type: Literal['2d', '3d']='2d',
                     intra_cutoff: float=5.0):

    """
    Main data preparation function
    
    """

    if isinstance(data, mol.LigandData):
        ligand = True

    elif isinstance(data, mol.ProteinData):
        ligand = False

    else:
        raise ValueError('Invalid schema')


    # Preparing Atom (Node) features and Positions
    data.atom_features, data.positions = construct_feature_matrix_jax(
                                            data.mol,
                                            ligand,
                                            features.ligand_features,
                                            features.protein_features)


    # Preparing Edge Index

    if graph_type == '2d':
        data.edge_index = construct_edge_matrix_2d_jax(data.mol,
                                                undirected=undirected,
                                                self_loop=self_loop)

    elif graph_type == '3d':
        data.edge_index = construct_edge_matrix_3d_jax(data.positions,
                                                   intra_cutoff=intra_cutoff)

    else:
        raise ValueError(f'Unrecognized Option for Type of Graph: {graph_type}')


    return data



def make_graph(data: mol.LigandData | mol.ProteinData,
               undirected=True,
               self_loop=False,
               features: mol.Features = None,
               graph_type: Literal['2d', '3d']='2d',
               intra_cutoff: float=5.0) -> mol.MolGraph:

    mol_graph = mol.MolGraph()

    features = features if features is not None else mol.Features()

    data = prepare_mol_data(data, 
                            undirected=undirected,
                            self_loop=self_loop,
                            features=features,
                            graph_type=graph_type,
                            intra_cutoff=intra_cutoff)


    mol_graph.x = data.atom_features
    mol_graph.pos = data.positions
    mol_graph.edge_index = data.edge_index


    return mol_graph

    

def combine_graphs(ligand_graph: mol.MolGraph,
                   protein_graph: mol.MolGraph,
                   add_interaction_edges: bool=False,
                   edge_type: bool=False,
                   inter_cutoff: int | float=5) -> mol.MolGraph:

    mol_graph = mol.MolGraph()
    interaction_edges = None


    mol_graph.x = jnp.concatenate([ligand_graph.x,
                   protein_graph.x],
                   axis=0)



    mol_graph.pos = jnp.concatenate([ligand_graph.pos,
                     protein_graph.pos],
                     axis=0)

    mol_graph.mol_id = add_ligand_protein_id_jax(
        ligand_graph.x,
        protein_graph.x
    )


    if add_interaction_edges:
        interaction_edges = construct_interaction_edges_jax(ligand_graph.pos,
                                                        protein_graph.pos,
                                                        cutoff=inter_cutoff)


    mol_graph.edge_index = combine_edge_index_jax(ligand_graph.edge_index,
                                    protein_graph.edge_index,
                                    ligand_graph.x.shape[0],
                                    interaction_edges)


    # 0 -> ligand bonds; 1 -> protein bonds -> 2 interaction between ligand and protein

    if edge_type:
        mol_graph.edge_type = add_edge_type_jax(ligand_graph.edge_index,
                              protein_graph.edge_index,
                              interaction_edges)


    return mol_graph



key = random.key(42)


lig_x = random.normal(key, shape=(10, 32))
prot_x = random.normal(key, shape=(50, 32))


add_ligand_protein_id_jax(lig_x,
                          prot_x)