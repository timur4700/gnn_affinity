from pathlib import Path
from utils import general
import re
import os
from typing import Any

import shutil

def find_metadata(dataset_directory: Path) -> dict[str, Any] | None:
    pattern = r'metadata'

    for file in os.listdir(dataset_directory):
        if re.findall(pattern=pattern,
                      string=file):

            return general.load_json(dataset_directory/file)

    return None


    

def check_metadata(metadata: dict[str, Any],
                   model_name: str) -> bool:

    if metadata is None:
        print("Dataset's metadata not found")
        return False

    print(f'Metadata successfully found')

    dataset_model = metadata["model"]

    if dataset_model != model_name:
        print(f'The dataset does not fit with {model_name} model')
        return False

    print(f'Model fits to prepared dataset')

    dataset_path = metadata['dataset_path']

    if not os.path.exists(dataset_path):
        print(f"The dataset stated in metadata was not found")

    print('Dataset was found')

    return True





def make_model_directory(parent_directory: Path,
                         id: str):

    model_directory_path = parent_directory / f'model_{id}'
    model_params_directory = model_directory_path / 'parameters'

    os.makedirs(model_directory_path, exist_ok=True)
    os.makedirs(model_params_directory, exist_ok=True)

    return {'main': model_directory_path,
            'params': model_params_directory}



def model_config(model_directory: Path,
                 model_name: str,
                   id: str) -> Path:

    model_config_source = Path(__file__).parent.parent / 'built_in_models' / 'all_settings' / f'{model_name}.yaml'
    target_config_destination = model_directory / f'model_settings_{id}.yaml'

    shutil.copy(model_config_source,
                target_config_destination)

    return target_config_destination



def trainer_config(model_directory: Path,
                   id: str) -> Path:

    trainer_config_source = Path(__file__).parent.parent.parent / 'train' / 'settings.yaml'
    target_config_destination = model_directory / f'train_settings_{id}.yaml'

    shutil.copy(trainer_config_source,
                target_config_destination)

    return target_config_destination
    

    


    



    




    

