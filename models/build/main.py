from pathlib import Path
from models.build import utils
from models.build import model_register
from models.build import metadata


def build_model(dataset_directory: Path,
                model_name: str):

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
    model_params = model_data.custom_params['ModelSettings']

    model_metadata = metadata.ModelMetaData(model_name=model_name,
                                       id=dataset_id,
                                       model_params=model_params,
                                       dataset_metadata=dataset_metadata,
                                       model_saved_params=str(model_directory['params']))



    model_metadata.model_config_path = str(utils.model_config_save(
                                                    model_directory['main'],
                                                    model_name,
                                                    dataset_id).resolve())

    model_metadata.trainer_config_path = str(utils.trainer_config_save(
                                                    model_directory['main'],
                                                    dataset_id).resolve())


    model_metadata.save(metadata_path)

    print(f"\nModel main directory: {model_directory['main'].resolve()}")
    print(f'Model Configuration Path: {model_metadata.model_config_path}')
    print(f"Train Configuration Path: {model_metadata.trainer_config_path}")
    print(f"Dataset directory: {dataset_directory.resolve()}")