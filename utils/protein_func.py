from rdkit import Chem
from typing import Literal
from utils import general, chem

import numpy as np


def get_residue_ids(protein: Chem.Mol) -> np.ndarray:

    return np.array([atom.GetPDBResidueInfo().GetResidueNumber() for atom 
                     in protein.GetAtoms()], dtype=int)



def pocket_extraction(ligand: Chem.Mol,
                      protein: Chem.Mol,
                      cutoff_distance: float=10,
                      method: Literal['cog', 'com', 'atom']='atom',
                      sanitize=False):

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

    d_ij = chem.get_distance(positions, protein_positions)

    cutoff_mask = d_ij <= cutoff_distance
    protein_atoms_in_cutoff = np.any(cutoff_mask, axis=0)
    protein_residues_in_cutoff = np.unique(resids[protein_atoms_in_cutoff])


    atom_ids2extract = np.nonzero(np.isin(resids, 
                                          protein_residues_in_cutoff))[0]


    extracted_pocket = chem.mol_subset(protein, atom_ids2extract)

    if sanitize:
        Chem.SanitizeMol(extracted_pocket)

    return extracted_pocket