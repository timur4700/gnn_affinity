import os
from typing import Sequence, Any

from pathlib import Path

import pandas as pd
from datasets.pdbbind import worker, metadata
from utils import general, mol_features, schemas

from models.build import model_register

from multiprocessing import Pool
from tqdm import tqdm
import time

import pickle

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

def index_preprocess(index_data: pd.DataFrame) -> dict[str, list[Any]]:

    return (index_data.set_index('pdb_id')[['-logKd/Ki', 'dataset']]
            .rename(columns={'-logKd/Ki': 'y'})
            .to_dict('index')
            )




def chunk_size_calc(iterable: Sequence[Any]) -> int:
    n_proc = os.cpu_count()
    chunk, extra = divmod(len(iterable), n_proc * 4)

    if extra:
        chunk += extra

    return chunk



def init_worker(index_data: pd.DataFrame, 
                config: dict[str, str|float], 
                features: schemas.Features):

    general.limit_native_threads()
    
    global _index_data, _config, _features
    _index_data = index_preprocess(index_data)
    _config = config
    _features = features


def main_worker(pdb_dir):
        try:
            work = worker.PDBWorker(pdb_dir,
                                    _index_data,
                                    _config,
                                    _features,
                                    sanitize=False)

            result = work.init_graph_preparation()
            return result, None

        except Exception as e:
             return None, str(pdb_dir.name)



def preprocess_dataset(pdb_directory_path: Path,
         index_data_path: Path,
         destination_path: Path,
         model_name=None):

    # Generating unique dataset_id
    dataset_id = general.make_unique_id()

    # Loading YAML configuration file
    config_path = Path(__file__).resolve().parent / 'config.yaml'
    config = general.load_yaml(config_path)

    if model_name is None:
        print('The model was not provided\nAll available features will be preprocessed')


    # Loading Feature Extraction Functions
    model = model_register.MODEL_REGISTER.get(model_name, None)


    if model is None:
        print(f"The model {model_name} was not found in the register")
        print('Preparing all available features')
        features = schemas.Features(
             ligand_features=mol_features.get_all_ligand_features(),
             protein_features=mol_features.get_all_protein_features()
        )

    else:
         features = model().features


    # Defining Errors Log
    failed_pdb = list()
    log_file_header = "Unprocessed PDBs:\n"

    # Defining INDEX PDBBind Data (v.2020)
    index_data = pd.read_csv(index_data_path)


    # Constructing paths to PDB entries
    paths = [pdb_directory_path / pdb for pdb in os.listdir(pdb_directory_path)]
    destination_path = destination_path / f'prepared_data_{dataset_id}'

    # Making dataset directory in the target directory
    os.makedirs(destination_path, exist_ok=True)

    # Defining dataset path
    dataset_path = destination_path / f'pdbbind_graph_dataset_{dataset_id}.pkl'

    # Defining metadata path
    metadata_path = destination_path / f'metadata_{dataset_id}.json'

    # Obtaining CPU count on a machine
    n_proc = os.cpu_count()

    # Configuring CHUNCK size for each working during MP dataset preprocessing
    if config['mol_preparation']['protein']['type'] == 'protein':
        chunk_size = 100

    else:
        chunk_size = chunk_size_calc(paths)

    start = time.perf_counter()


    # Dataset Metadata
    pdb_metadata = metadata.PDBdatasetMeta(
                        model=model_name,
                        id=dataset_id,
                        graph_config=metadata.make_graph_config(features,
                                                                config),
                        dataset_path=dataset_path)

    # Main Preparation block
    with open(dataset_path, 'wb') as f:
        graphs_prepared = 0

        with Pool(n_proc, init_worker, (index_data, 
                                        config, 
                                        features)) as pool:
                
                try:
                    for result, error in tqdm(pool.imap(main_worker, 
                                                paths,
                                                chunksize=chunk_size), 
                                    total=len(paths)):
        
                        if error is not None:
                            failed_pdb.append(error)
                            continue

                        pickle.dump(result, f)
                        graphs_prepared += 1

        
                finally:
                    pool.close()
                    pool.join()
                    finish = time.perf_counter() - start
                    print(f"Execution time: {finish} s")


                    num_errors = len(failed_pdb)
                    error_log = log_file_header + '\n'.join(failed_pdb) + f"\nTotal Errors: {num_errors}"

                    with open(destination_path / f'error_{dataset_id}.log', 'w') as f:
                        f.write(error_log)

                    pdb_metadata.graph_num = graphs_prepared
                    pdb_metadata.status = 'completed'
                    pdb_metadata.save(metadata_path)

                    print(f"PDBbind Graph Dataset was successfully saved in {str(destination_path)}")