from pathlib import Path
from models.build import utils
from registers.models.models import MODEL_REGISTER
from metadata.model import ModelMetaData
from metadata.graphs import GraphDatasetMeta

from schemas.general import find_metadata


def build_model(graph_dataset_directory: Path):


    graph_dataset_metadata: GraphDatasetMeta = find_metadata(graph_dataset_directory,
                                                             metadata=GraphDatasetMeta)



    dataset_parent_directory = graph_dataset_metadata.dataset_path.parent

    model_name = graph_dataset_metadata.model
    dataset_id = graph_dataset_metadata.id


    model_factory = MODEL_REGISTER.get(model_name, None)   

    if not model_factory:
        print(f'The requested model {model_name} was not found in the registry')
        print('Closing the program')
        return

    model_directory = utils.make_model_directory(dataset_parent_directory, 
                                                 dataset_id)

    metadata_path = model_directory['main'] / f'metadata_{dataset_id}.json'

    model_data = model_factory()
    model_params = model_data.custom_params['ModelSettings']

    model_metadata = ModelMetaData(model_name=model_name,
                                   id=dataset_id,
                                   model_params=model_params,
                                   dataset_metadata=graph_dataset_metadata,
                                   model_saved_params=model_directory['params'])



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
    print(f"Dataset directory: {graph_dataset_metadata.dataset_path.resolve()}")