from pathlib import Path

from utils import general
from configs.train import TrainLoaderConfig

from metadata.graphs import GraphDatasetMeta
from metadata.model import ModelMetaData, RunPaths
from metadata.train import RunMetaData

from train import trainer
from train import helpers
from train.utils import check_device

from registers import spliters
from registers.models.models import MODEL_REGISTER

from torch.nn import Module



def load_train_configs(trainer_config_path: Path):

    train_loader_configs = general.load_yaml(
        trainer_config_path
    )

    return TrainLoaderConfig.load_data(
        train_loader_configs
    )


def configure_trainer(
        trainer: trainer.Trainer,
        model: Module,
        run_metadata: RunMetaData,
        dataset: list,
        splitter,
        batch_size,
        mode='new',
        checkpoint_path: Path | None=None,
        model_weights: Path | None=None
) -> None:

    trainer.set_model(
        model,
        model_checkpoint=checkpoint_path,
        mode=mode
    ) 
    
    trainer.set_paths(
        run_metadata
    )
    
    trainer.set_dataset(
        dataset,
        splitter,
        batch_size
    )


def prepare_run_data(
        model_metadata: ModelMetaData
) -> tuple[RunMetaData, Path]:
    
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

    return run_metadata, run_metadata_path


def build_trainer(
        run_metadata: RunMetaData
) -> tuple[trainer.Trainer, TrainLoaderConfig]:

    train_configs = load_train_configs(
        run_metadata.train_config_path
        )

    train_configs.train.device = check_device(
        train_configs.train.device
    )

    model_trainer = trainer.Trainer(
        **train_configs.train.model_dump()
    )

    return (model_trainer, train_configs)


def load_dataset_with_splitter(
        model_metadata: ModelMetaData
):

    dataset_metadata = GraphDatasetMeta.load_data(
        model_metadata.dataset_metadata
    )

    dataset_path = dataset_metadata.dataset_path
    dataset = general.unpack_pickle(dataset_path)

    splitter = spliters.SPLITERS.get(
        dataset_metadata.name,
        spliters.SPLITERS['default']
    )

    return (dataset, splitter)


def build_model(
        model_metadata: ModelMetaData,
        run_metadata: RunMetaData,
        train_configs: TrainLoaderConfig
):

    model_name = model_metadata.model_name
    model_config = general.load_yaml(
            run_metadata.model_config_path)['ModelSettings']
    
    model = MODEL_REGISTER.get(model_name, None)

    if model is None:
        raise ValueError(f'Model {model_name} is not registered')
        
    model: Module = model().model_class()(**model_config).to(
            train_configs.train.device
        )

    return model



def collect_train_results(
        trainer: trainer.Trainer,
        run_metadata: RunMetaData
) -> RunMetaData:

    metrics = trainer.predict_test(trainer.best_model_val_loss_param)
    run_metadata.evaluation_metrics = metrics

    return run_metadata    


def runs_human_readble(
        runs: list[RunPaths]
):
    return [
        f"{run.name.replace('_', ' ')}" for run in runs
    ]


def choosing_model_run(
        metadata: ModelMetaData
) -> RunPaths:

    model_name = metadata.model_name
    model_id = metadata.id

    runs = runs_human_readble(metadata.model_runs)
    choice = general.mco(
        f"Choose Run for {model_name}, ID {model_id}:",
        runs
    )

    return metadata.model_runs[choice]
