from utils.chem import AtomFeatureExtract, ProteinFeatureExtract
from collections.abc import Callable


def get_all_ligand_features() -> dict[str, Callable]:

    return {
        'atomic_number': AtomFeatureExtract.get_atom_type,
        'aromatic_status': AtomFeatureExtract.get_aromatic_type,
        'num_hydrogens': AtomFeatureExtract.get_num_h_bonds
    }


def get_all_protein_features():
    protein_features = get_all_ligand_features()

    protein_features.update({'residue_type': ProteinFeatureExtract.get_residue_name})
    return protein_features