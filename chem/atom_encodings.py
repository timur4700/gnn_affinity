RESIDUES = {
    "ALA": 1,  # Alanine
    "ARG": 2,  # Arginine
    "ASN": 3,  # Asparagine
    "ASP": 4,  # Aspartic acid
    "CYS": 5,  # Cysteine
    "GLN": 6,  # Glutamine
    "GLU": 7,  # Glutamic acid
    "GLY": 8,  # Glycine
    "HIS": 9,  # Histidine
    "ILE": 10,  # Isoleucine
    "LEU": 11, # Leucine
    "LYS": 12, # Lysine
    "MET": 13, # Methionine
    "PHE": 14, # Phenylalanine
    "PRO": 15, # Proline
    "SER": 16, # Serine
    "THR": 17, # Threonine
    "TRP": 18, # Tryptophan
    "TYR": 19, # Tyrosine
    "VAL": 20, # Valine,
    'Other': 21
}


ATOMS = {
    'H':  1,
    'B': 5,
    'C':  6,
    'N':  7,
    'O':  8,
    'P': 15,
    'S': 16,
    'Se': 34,
    'halogens': [9, 17, 35, 53],
    'metals': (
        [3, 4, 11, 12, 13] + 
        list(range(19, 32)) +
        list(range(37, 51)) + 
        list(range(55, 84)) +
        list(range(87, 104))
    )
}


ATOMS_MAPPING = {}


for i, (k, v) in enumerate(ATOMS.items()):

    if (k == 'halogens' or
        k == 'metals'):

        for atomic_num in v:
            ATOMS_MAPPING[atomic_num] = i

        continue

    ATOMS_MAPPING[v] = i