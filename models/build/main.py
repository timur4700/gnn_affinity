from pathlib import Path
from models.build import utils
from models.build import model_register
from models.build import metadata

from dataclasses import asdict

import torch
from torch.nn import Module


def build_model(dataset_directory: Path,
                model_name: str,
                default_params: bool=True):

    dataset_parent_directory = dataset_directory.parent
    dataset_metadata = utils.find_metadata(dataset_directory)

    dataset_id = dataset_metadata['id']

    if not utils.check_metadata(dataset_metadata, 
                                model_name):
        print('Closing the program')
        return

    model_factory = model_register.MODEL_REGISTER.get(model_name,
                                                      None)   

    if not model_factory:
        print(f'The requested model {model_name} was not found in the registry')
        print('Closing the program')
        return

    model_directory = utils.make_model_directory(dataset_parent_directory, 
                               dataset_id)

    metadata_path = model_directory['main'] / f'metadata_{dataset_id}.json'

    model_data = model_factory()
    model_params = asdict(model_data.default_params) if default_params else model_data.custom_params['ModelSettings']
    model: Module = model_data.model_class(**model_params)

    init_model_params_path = model_directory['main'] / 'init_model.pth'


    model_metadata = metadata.MetaData(model_name=model_name,
                                       id=dataset_id,
                                       model_params=model_params,
                                       dataset_metadata=dataset_metadata,
                                       model_init_params_path=str(init_model_params_path),
                                       model_saved_params=str(model_directory['params']))

    model_metadata.trainer_config_path = str(utils.trainer_config(model_directory['main'],
                                                              dataset_id))


    torch.save(model.state_dict(), init_model_params_path)
    model_metadata.save(metadata_path)

    print(f"Model main directory: {model_directory['main'].resolve()}")
    print(f"Dataset directory: {dataset_directory.resolve()}")