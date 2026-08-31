import os
import tarfile
from tqdm import tqdm
from datasets.pdbbind import index_data_prep
from datasets import metadata

import shutil
import re

from utils import general

from pathlib import Path

# Only PDBbind v.2020 + CASF2016

gen_pattern = '[-,_*\s]*'
general_set = rf'\bpdbbind{gen_pattern}v2020{gen_pattern}other{gen_pattern}pl\b'
refined_set = rf'\bpdbbind{gen_pattern}v2020{gen_pattern}refined\b'


folder_names = {
    general_set: 'general-set',
    refined_set: 'refined_set'
}



# Molecule Preparation Configuration File
mol_config_path = Path(__file__).resolve().parent / 'config.yaml'




def tar_extract(directory_path: Path) -> dict[str, Path]:

    paths = {}

    for compr_folder in os.listdir(directory_path):

        if compr_folder.endswith('.tar.gz'):
            compr_folder = compr_folder.lower()
            for pattern, name in folder_names.items():

                if re.findall(pattern, compr_folder):                
                    source = directory_path / compr_folder
                    destination = directory_path / name
                    os.makedirs(destination, exist_ok=True)

                    with tarfile.open(source, 'r:*') as tar:
                        tar.extractall(destination)

                    print(f"The PDBBind {name} found --> extracted to {str(destination)}")

                    paths[name] = destination / os.listdir(destination)[0]

    return paths


def merge_folders(directory_path: Path,
                  folders: dict[str, Path],
                  metadata: metadata.DatasetMetadata):

    entry_name = 'pdbbind_2020_entries'
    entries_path = directory_path / entry_name
    file_paths = dict()

    for name, folder_path in folders.items():
        folder_list = os.listdir(folder_path)
        for pdb_entry in tqdm(folder_list, 
                              desc=f'Transfering PDBs from {name}'):

            if pdb_entry == 'readme':
                continue


            pdb_source_path = folder_path / pdb_entry
            pdb_dest_path = entries_path / pdb_entry
            shutil.copytree(pdb_source_path, pdb_dest_path, 
                            dirs_exist_ok=True)
            
            file_paths[pdb_entry] = pdb_dest_path


        entries_path_str = str(entries_path.resolve())

        print(f'All PDB entries in {name} were transfered to {entries_path_str}')


    metadata.entries_path = entries_path_str

    return file_paths



def main(directory_path: Path):

    dataset_metadata = metadata.DatasetMeta(
        name='pdbbind'
    )


    folder_paths = tar_extract(directory_path)
    merged_file_paths = merge_folders(directory_path,
                                      folder_paths,
                                      dataset_metadata)


    csv_data_path = directory_path / 'pdbbind_data.csv'
    index_dir_path = merged_file_paths['index']
    index_data_prep.main(index_dir_path).to_csv(csv_data_path)

    csv_data_path = str(csv_data_path.resolve())
    dataset_metadata.target_path = csv_data_path
    dataset_metadata.save(directory_path / 'metadata.json')

    mol_config = general.load_yaml(mol_config_path)

    print(f'File with target variable saved in  {csv_data_path}') 
