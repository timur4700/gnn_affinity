from torch.nn import Module
from registers import spliters
from train import trainer
from utils import general

from pathlib import Path

from typing import Any

from registers.models.models import MODEL_REGISTER

from train.utils import save_model_run, check_device
from configs import train

from dataclasses import asdict


def start_trainer(model_metadata: dict[str, Any],
                  config: dict[str, Any]):


    trainer_config = train.TrainConfig(**config['TrainingSettings'])
    loader_config = train.LoaderConfig(**config['LoaderSettings'])

    trainer_config.device = check_device(trainer_config.device)

    dataset_metadata = model_metadata['dataset_metadata']
    dataset_path = Path(dataset_metadata['dataset_path'])

    dataset = general.unpack_pickle(dataset_path)

    spliter = spliters.SPLITERS.get(dataset_metadata['name'],
                                    spliters.SPLITERS['default'])

    model_name = model_metadata['model_name']
    model_config = general.load_yaml(model_metadata['model_config_path'])['ModelSettings']

    model = MODEL_REGISTER.get(model_name, None)


    if not model:
        print('Model not found\nClose...')

    model: Module = model().model_class()(**model_config).to(trainer_config.device)


    model_trainer = trainer.Trainer(**asdict(trainer_config))
    model_trainer.set_model(model,
                            Path(model_metadata['model_saved_params']))

    model_trainer.set_dataset(dataset,
                              spliter,
                              loader_config.batch_size)


    model_trainer.start_train()
    metrics = model_trainer.predict()

    save_model_run(model_metadata,
                   model_trainer.best_model_val_loss_param,
                   model_config,
                   config,
                   metrics,
                   model_trainer.train_log)