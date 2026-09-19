from train import helpers
from metadata.model import ModelMetaData
from train.workflows import utils



def new_training(
        model_metadata: ModelMetaData
) -> ModelMetaData:


    run_metadata, run_metadata_path = utils.prepare_run_data(
        model_metadata
    )

    model_trainer, train_configs = utils.build_trainer(
        run_metadata
    )

    dataset, splitter = utils.load_dataset_with_splitter(
        model_metadata
    )

    model = utils.build_model(
        model_metadata,
        run_metadata,
        train_configs
    )

    utils.configure_trainer(
        model_trainer,
        model,
        run_metadata,
        dataset,
        splitter,
        train_configs.loader.batch_size,
        run_metadata.model_checkpoint_path
    )

    wrapped_train = helpers.status_wrapper(
        model_trainer.start_train,
        model_trainer,
        run_metadata,
        run_metadata_path
    )

    wrapped_train()

    run_metadata = utils.collect_train_results(
        model_trainer, 
        run_metadata
    )

    run_metadata.save(
        run_metadata_path
    )

    return model_metadata