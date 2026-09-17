from torch.nn import Module
from registers import spliters
from train import trainer
from utils import general

from pathlib import Path

from typing import Any

from registers.models.models import MODEL_REGISTER
from metadata.model import ModelMetaData
from metadata.graphs import GraphDatasetMeta
from metadata.datasets import DatasetMetadata

from train.utils import save_model_run, check_device
from configs import train

from metadata.base import find_metadata

from metadata.train import RunMetaData
from train import helpers





def load_train_configs(model_metadata: ModelMetaData):

    train_loader_configs = general.load_yaml(
        model_metadata.trainer_config_path
    )

    return train.TrainLoaderConfig.load_data(
        train_loader_configs
    )

def start_trainer(
        model_directory: Path,
        retrain: bool=False,
        model_params=None
):

    model_metadata: ModelMetaData = find_metadata(model_directory,
                                                  ModelMetaData)

    model_id = model_metadata.id
    run_id, run_directory = helpers.make_run_directory(
        model_metadata.model_saved_params
    )

    run_metadata = helpers.make_run_metadata(
        model_metadata,
        run_id,
        run_directory
    )

    run_metadata_path = (
        run_directory /
        'metadata.json'
    )

    helpers.transfer_configs2run_directory(
        model_metadata,
        run_metadata
    )


    train_configs = load_train_configs(model_metadata)


    train_configs.train.device = check_device(
        train_configs.train.device
    )

    dataset_metadata = GraphDatasetMeta.load_data(
        model_metadata.dataset_metadata
    )

    dataset_path = dataset_metadata.dataset_path
    dataset = general.unpack_pickle(dataset_path)
    
    spliter = spliters.SPLITERS.get(
        dataset_metadata.name,
        spliters.SPLITERS['default']
    )

    model_name = model_metadata.model_name
    model_config = general.load_yaml(
        model_metadata.model_config_path
    )['ModelSettings']


    model = MODEL_REGISTER.get(model_name, None)
    
    model: Module = model().model_class()(
        **model_config
    ).to(
        train_configs.train.device
    )


    model_trainer = trainer.Trainer(
        **train_configs.train.model_dump()
    )

    model_trainer.set_model(
        model, 
        model_metadata.model_saved_params
    )

    model_trainer.set_paths(
        run_metadata
    )

    model_trainer.set_dataset(
        dataset,
        spliter,
        train_configs.loader.batch_size
    )

    helpers.save_and_update_status(
        run_metadata,
        'training',
        run_metadata_path
    )

    wrapped_train = helpers.status_wrapper(
        model_trainer.start_train,
        run_metadata,
        run_metadata_path
    )

    wrapped_train()

    #model_trainer.start_train()
    metrics = model_trainer.predict_test(model_trainer.best_model_val_loss_param)

    run_metadata.evaluation_metrics = metrics

    helpers.save_and_update_status(
        run_metadata,
        'finished',
        run_metadata_path
    )

    run_metadata.save(
        run_metadata_path
    )