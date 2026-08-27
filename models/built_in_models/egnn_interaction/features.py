from utils import schemas
from utils.chem import AtomFeatureExtract, ProteinFeatureExtract




def egnn_interaction_features():

    features = {
        'atomic_num': AtomFeatureExtract.get_atom_type,
        'aromatic_status': AtomFeatureExtract.get_aromatic_type,
        'num_hudrogens': AtomFeatureExtract.get_num_h_bonds
    }

    return schemas.Features(features, features)