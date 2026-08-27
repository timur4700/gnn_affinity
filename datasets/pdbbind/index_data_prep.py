import pandas as pd
from pathlib import Path
import json
import re


SCRIPT_DIR = Path(__file__).resolve().parent


def dataset_mapping(datasets: dict[str, list[str]]):

    mapping = {
        i:k for k, v in datasets.items() for i in v
    }

    return mapping



def affinity_data_prep(pdb_affinity_data: Path) -> pd.DataFrame:
# PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name
    cols = ['pdb_id', 
            'resolution', 
            'year', 
            'dataset', 
            '-logKd/Ki',
            'ligand_name']


    with open(SCRIPT_DIR / 'pdbbind_datasets.json') as f:
        datasets = json.load(f)

    mapping = dataset_mapping(datasets)

    rows = []




    with open(pdb_affinity_data) as f:
        while True:
            line = f.readline().strip()

            if not line:
                break

            if not line.startswith('#'):
                line = line.split()

                pdb_id = line[0]
                resolution = float(line[1]) if line[1] != 'NMR' else None
                year = int(line[2])
                dataset = mapping.get(pdb_id, None)
                ki_kd = float(line[3])
                ligand_name = re.findall(r"\(([^()]*)\)", line[-1])

                if ligand_name:
                    ligand_name = ligand_name[0]

                else:
                    ligand_name = None

                rows.append([pdb_id,
                       resolution,
                       year,
                       dataset,
                       ki_kd,
                       ligand_name])


    return pd.DataFrame(rows, columns=cols)



def protein_name_prep(pdb_protein_name: Path) -> pd.DataFrame:

    cols = ['pdb_id',
            'uniprot_id',
            'protein_name']

    rows = []


    with open(pdb_protein_name) as f:
        while True:
            line = f.readline().strip()

            if not line:
                break

            if not line.startswith('#'):
                line = line.split()

                pdb_id = line[0]
                uniprot_id = line[2] if line[2] != '------' else None
                prot_name = '_'.join(line[3:])

                rows.append([pdb_id,
                            uniprot_id,
                            prot_name])

    return pd.DataFrame(rows, columns=cols)




def main(index_dir_path: Path):

    pdb_affinity_data = index_dir_path / 'INDEX_general_PL_data.2020'
    pdb_protein_name = index_dir_path / 'INDEX_general_PL_name.2020'


    affinity_df = affinity_data_prep(pdb_affinity_data)
    protein_name_df = protein_name_prep(pdb_protein_name)

    merged_df = pd.merge(affinity_df, protein_name_df, on='pdb_id')

    return merged_df
