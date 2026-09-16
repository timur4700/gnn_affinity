import jax.numpy as jnp
from jax import jit
from rdkit import Chem



def _adjacency_rdkit_jax(mol: Chem.Mol,
                     a_matrix: jnp.ndarray,
                     undirected: bool=True) -> None:

    for bond in mol.GetBonds():
            i = bond.GetBeginAtom().GetIdx()
            j = bond.GetEndAtom().GetIdx()
    
            a_matrix[i, j] = 1
    
            if undirected:
                a_matrix[j, i] = 1



def adjacency_matrix_jax(u_mol: Chem.Mol,
                     undirected: bool=True,
                     self_loop: bool=False) -> jnp.ndarray:
    """
    Constructs Adjacency Matrix (A), based on atom connectivity from 
    MDAnalysis.Universe Object or rdkit.Chem.Mol

    Parameters
    ----------
    u : mda.Universe, or rdkit.Chem
        Universe/Mol object

    undirected : bool
        If turned on, makes undirected edge index list.
        Default True

    self_loop : bool
        If turned on adds self-loop to each atom.
        Default False

    """

    n_atoms = u_mol.GetNumAtoms()
    a_matrix = jnp.zeros((n_atoms, n_atoms))

    if isinstance(u_mol, Chem.Mol):
         _adjacency_rdkit_jax(u_mol, a_matrix, undirected)

    else:
        raise ValueError

    if self_loop:
        jnp.fill_diagonal(a_matrix, 1)


    return a_matrix



def get_residue_ids_jax(protein: Chem.Mol) -> jnp.ndarray:

    resids = jnp.zeros(protein.GetNumAtoms())

    for i, atom in enumerate(protein.GetAtoms()):
        residue_id = atom.GetPDBResidueInfo().GetResidueNumber()
        resids[i] = residue_id


    return resids







# Jit
@jit
def get_distance_jax(positions_a: jnp.ndarray,
                 positions_b: jnp.ndarray) -> jnp.ndarray:

    r_ij = positions_a[:,None,:] - positions_b[None,:,:]
    d_ij = jnp.sqrt(jnp.sum(r_ij**2,axis=-1))

    return d_ij




@jit
def to_sparse_adjacency_matrix_jax(a_matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Transformse dense Adjacency Matrix (A) to sparse matrix (edge index list) with shape [2, E],
    where E is the number of edges

    Parameters
    ----------
    a_matrix : np.ndarray
        Dense Adjacency Matrix


    Returns
    -------
    np.ndarray
        Sparse Adjacency Matrix
    """


    edge_index = jnp.array(jnp.nonzero(a_matrix))

    return edge_index