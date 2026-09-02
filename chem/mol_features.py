from rdkit.Chem import Atom, Bond
from chem import atom_encodings

import inspect
from collections.abc import Callable



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