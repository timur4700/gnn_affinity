from rdkit import Chem
from rdkit.Chem import Atom, Bond

import MDAnalysis as mda

from utils import atom_encodings
import inspect

from pathlib import Path
from typing import Literal

import numpy as np
import torch




class AtomFeatureExtract:
    @staticmethod
    def get_atom_type(atom: Atom) -> int:
        return atom.GetAtomicNum()

    @staticmethod
    def get_num_h_bonds(atom: Atom) -> int:
        return atom.GetTotalNumHs()

    @staticmethod
    def get_aromatic_type(atom: Atom) -> int:
        return 1 if atom.GetIsAromatic() else 0


    def extract_func(self):

        return {
                name: method
                for name, method in inspect.getmembers(
                    self,
                    predicate=inspect.isfunction
                )
                if not name.startswith("_")}


class ProteinFeatureExtract(AtomFeatureExtract):

    @staticmethod
    def get_residue_name(atom: Atom) -> int:

        resname = atom.GetPDBResidueInfo().GetResidueName()
        resid = atom_encodings.RESIDUES.get(resname, atom_encodings.RESIDUES["Other"])

        return resid





class BondFeatureExtract:

    @staticmethod
    def get_is_bond_aromatic(bond: Bond) -> int:
       return 1 if bond.GetIsAromatic() else 0


    @staticmethod
    def get_stereotype(bond: Bond) -> int:
        return None



def _adjacency_mda(u: mda.Universe,
               a_matrix: np.ndarray,
               undirected: bool=True) -> None:

    for atom_a, atom_b in u.bonds:
            i = atom_a.index
            j = atom_b.index
    
            a_matrix[i, j] = 1
    
            if undirected:
                a_matrix[j, i] = 1



def _adjacency_rdkit(mol: Chem.Mol,
                     a_matrix: np.ndarray,
                     undirected: bool=True) -> None:

    for bond in mol.GetBonds():
            i = bond.GetBeginAtom().GetIdx()
            j = bond.GetEndAtom().GetIdx()
    
            a_matrix[i, j] = 1
    
            if undirected:
                a_matrix[j, i] = 1



def adjacency_matrix(u_mol: mda.Universe | Chem.Mol,
                     undirected: bool=True,
                     self_loop: bool=False) -> np.ndarray:
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

    n_atoms = len(list(u_mol.atoms)) if isinstance(u_mol, mda.Universe) else u_mol.GetNumAtoms()
    a_matrix = np.zeros((n_atoms, n_atoms))

    if isinstance(u_mol, mda.Universe):
         _adjacency_mda(u_mol, a_matrix, undirected)

    elif isinstance(u_mol, Chem.Mol):
         _adjacency_rdkit(u_mol, a_matrix, undirected)

    else:
        raise ValueError

    if self_loop:
        np.fill_diagonal(a_matrix, 1)


    return a_matrix




def to_sparse_adjacency_matrix(a_matrix: np.ndarray) -> np.ndarray:
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


    edge_index = np.array(np.nonzero(a_matrix))

    return edge_index




def load_sdf(path: str | Path,
             sanitize=False) -> list[Chem.Mol]:

    supplier = Chem.SDMolSupplier(path, sanitize=sanitize)
    return [mol for mol in supplier if mol]



def load_mol(path: str | Path,
             sanitize=True) -> Chem.Mol:

    path = path if isinstance(path, Path) else Path(path)
    suffix = path.suffix

    rdkit_loaders = {
        '.mol2': Chem.MolFromMol2File,
        '.pdb': Chem.MolFromPDBFile
    }

    return rdkit_loaders[suffix](path, sanitize=sanitize)




def get_distance(positions_a: np.ndarray | torch.Tensor,
                 positions_b: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:

    r_ij = positions_a[:,None,:] - positions_b[None,:,:]

    if (isinstance(positions_a, torch.Tensor) and 
        isinstance(positions_b, torch.Tensor)):

        d_ij = torch.linalg.norm(r_ij, dim=-1)
        return d_ij

    d_ij = np.sum(r_ij**2,axis=-1)

    return d_ij




def get_residue_ids(protein: Chem.Mol) -> np.ndarray:

    resids = np.zeros(protein.GetNumAtoms())

    for i, atom in enumerate(protein.GetAtoms()):
        residue_id = atom.GetPDBResidueInfo().GetResidueNumber()
        resids[i] = residue_id


    return resids


def mol_subset(mol: Chem.Mol,
               remain_atoms: np.ndarray):
    atom_ids = set(remain_atoms)

    # find bonds where BOTH endpoints are in your atom set
    #bond_ids = [
    #    bond.GetIdx()
    #    for bond in mol.GetBonds()
    #    if bond.GetBeginAtomIdx() in atom_ids and bond.GetEndAtomIdx() in atom_ids
    #]

    bond_ids = set()
    for idx in atom_ids:
        for bond in mol.GetAtomWithIdx(int(idx)).GetBonds():
            if bond.GetOtherAtomIdx(int(idx)) in atom_ids:
                bond_ids.add(bond.GetIdx())
    bond_ids= list(bond_ids)

    atom_map = {}  # will map original_idx -> new submol_idx
    submol = Chem.PathToSubmol(mol, bond_ids, atomMap=atom_map)
    return submol


def mol_subset_old(mol: Chem.Mol,
               remain_atoms: np.ndarray) -> Chem.Mol:

    rw_mol = Chem.RWMol()
    old2new_idx = {}

    old_positions = mol.GetConformer().GetPositions()

    for i, atom in enumerate(mol.GetAtoms()):
        if i in remain_atoms:
             rw_mol.AddAtom(Chem.Atom(atom))
             old2new_idx[i] = len(old2new_idx)

    for bond in mol.GetBonds():

        i = bond.GetBeginAtom().GetIdx()
        j = bond.GetEndAtom().GetIdx()

        if i in remain_atoms and j in remain_atoms:
             rw_mol.AddBond(
                  old2new_idx[i],
                  old2new_idx[j],
                  bond.GetBondType()
             )

    new_conformer = Chem.Conformer(rw_mol.GetNumAtoms())

    for old, new in old2new_idx.items():
        new_conformer.SetAtomPosition(new, old_positions[old])


    rw_mol.AddConformer(new_conformer, assignId=True)
    
    extracted_mol = rw_mol.GetMol()
    #Chem.SanitizeMol(extracted_mol)

    return extracted_mol



def pocket_extraction(ligand: Chem.Mol,
                      protein: Chem.Mol,
                      cutoff_distance: float=10,
                      method: Literal['cog', 'com', 'atom']='atom'):

    positions = ligand.GetConformer().GetPositions()

    if method == 'atom':
        pass

    elif method == 'cog':
        positions = np.mean(positions, axis=0)[None,:]


    elif method == 'com':
         atom_masses = np.array([atom.GetMass() for atom in ligand.GetAtoms()])
         positions = np.average(positions, axis=0, weights=atom_masses)[None,:]

    else:
        raise ValueError(
        f"Unknown pocket extraction method: {method}"
                        )


    protein_positions = protein.GetConformer().GetPositions()

    resids = get_residue_ids(protein)

    d_ij = get_distance(positions, protein_positions)

    cutoff_mask = d_ij <= cutoff_distance
    protein_atoms_in_cutoff = np.any(cutoff_mask, axis=0)
    protein_residues_in_cutoff = np.unique(resids[protein_atoms_in_cutoff])

    atom_ids2extract = np.nonzero(np.isin(resids, 
                                          protein_residues_in_cutoff))[0]


    extracted_pocket = mol_subset(protein, atom_ids2extract)

    return extracted_pocket