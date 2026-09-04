from torch.nn import Module
from registers import spliters
from train import trainer
from utils import general

from pathlib import Path

from typing import Any

from registers.models.models import MODEL_REGISTER
from metadata.model import ModelMetaData
from metadata.graphs import GraphDatasetMeta

from train.utils import save_model_run, check_device
from configs import train

from dataclasses import asdict

from schemas.general import find_metadata



def load_train_configs(model_metadata: ModelMetaData):

    configs = dict()

    train_configs = general.load_yaml(
        model_metadata.trainer_config_path
        )

    configs['train'] = train.TrainConfig(
        **train_configs['TrainingSettings']
        )

    configs['loader'] = train.LoaderConfig(
        **train_configs['LoaderSettings']
    )

    return configs


def start_trainer(model_directory: Path,
                  retrain: bool=False,
                  model_params=None):


    model_metadata: ModelMetaData = find_metadata(model_directory,
                                                  ModelMetaData)

    train_configs = load_train_configs(model_metadata)


    train_configs['train'].device = check_device(train_configs['train'].device)

    dataset_metadata = GraphDatasetMeta(**model_metadata.dataset_metadata)
    dataset_path = dataset_metadata.dataset_path

    dataset = general.unpack_pickle(dataset_path)

    spliter = spliters.SPLITERS.get(dataset_metadata.name,
                                    spliters.SPLITERS['default'])

    model_name = model_metadata.model_name
    model_config = general.load_yaml(model_metadata.model_config_path)['ModelSettings']


    model = MODEL_REGISTER.get(model_name, None)
    model: Module = model().model_class()(**model_config).to(train_configs['train'].device)


    model_trainer = trainer.Trainer(**asdict(train_configs['train']))
    model_trainer.set_model(model, model_metadata.model_saved_params)
    model_trainer.set_dataset(dataset,
                              spliter,
                              train_configs['loader'].batch_size)


    model_trainer.start_train()
    metrics = model_trainer.predict_test(model_trainer.best_model_val_loss_param)

    save_model_run(model_metadata,
                   model_trainer.best_model_val_loss_param,
                   model_config,
                   train_configs,
                   metrics,
                   model_trainer.train_log)