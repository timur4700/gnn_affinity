from rdkit.Chem import Atom
import atom_encodings
import inspect


class AtomEncodings:

    
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





class ProteinsAtomEncoding(AtomEncodings):

    @staticmethod
    def get_residue_name(atom: Atom) -> int:

        resname = atom.GetPDBResidueInfo().GetResidueName()
        resid = atom_encodings.RESIDUES.get(resname, atom_encodings.RESIDUES["Other"])

        return resid
