from typing import Sequence, Any
import pandas as pd

from configs.preprocessing import PreprocessingData
from configs.mol import MolConfig

from registers.path_builders import PATHS_REGISTER
from registers.configs.mol import MOL_CONFIG
from registers.models.models import MODEL_REGISTER


from metadata.datasets import DatasetMetadata

from schemas.mol import Features

from utils import general

from chem.mol_features import (get_all_ligand_features,
                                get_all_protein_features)

from pathlib import Path



def get_model(model_name: str):
    model = MODEL_REGISTER.get(model_name)\

    return model()




def get_path_builder(dataset_metadata: DatasetMetadata,
                     preproc_data: PreprocessingData):
    
    name = dataset_metadata.name
    mol_config: MolConfig = preproc_data.configs.mol_config
    
    return PATHS_REGISTER.get(name).make(mol_config)



def index_preprocess(index_data: pd.DataFrame) -> dict[str, list[Any]]:

    return (index_data.set_index('pdb_id')[['-logKd/Ki', 'dataset']]
            .rename(columns={'-logKd/Ki': 'y'})
            .to_dict('index')
            )



def chunk_size_calc(iterable: Sequence[Any],
                    n_proc: int) -> int:
    chunk, extra = divmod(len(iterable), n_proc * 4)

    if extra:
        chunk += 1

    return chunk


def load_mol_config(config_path, 
                    dataset_name: str):
    
    raw_config = general.load_yaml(config_path)
    config_schema = MOL_CONFIG.get(dataset_name)

    return config_schema.load_data(raw_config['MolConfig'])



def load_config(config_path, 
                config_key: str, 
                schema):

    
    
    raw = general.load_yaml(config_path)
    return schema.load_data(raw[config_key])



def load_target_data(target_data_path: Path):

    target_data = pd.read_csv(target_data_path)
    target_data = index_preprocess(target_data)

    return target_data



def make_features(model_name) -> Features:

    if model_name is None:
        print('The model was not provided\n'\
              'All available features will be preprocessed')
    
        features = Features(
                            ligand_features=get_all_ligand_features(),
                            protein_features=get_all_protein_features()
                            )
    
    else:
        model = MODEL_REGISTER.get(model_name)
        features = model().features


    return features





def make_error_log(failed: list[str]):

    log_file_header = "Unprocessed PDBs:\n"
    num_errors = len(failed)

    error_log = log_file_header + \
                    '\n'.join(failed) + \
                        f"\nTotal Errors: {num_errors}"

    return error_log







