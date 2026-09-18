from train import helpers
from metadata.model import ModelMetaData
from train.workflows import utils

from metadata.train import RunMetaData


def resume_training(
        model_metadata: ModelMetaData
) -> ModelMetaData:


    run_data = utils.choosing_model_run(
        metadata=model_metadata
    )

    run_metadata_path = run_data.path

    run_metadata = RunMetaData.load_from_file(
        run_data.path
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
        'resume',
        run_metadata.model_checkpoint_path,
    )

    wrapped_train = helpers.status_wrapper(
        model_trainer.start_train,
        model_trainer,
        run_metadata,
        run_metadata_path
    )

    wrapped_train()

    metrics = model_trainer.predict_test(model_trainer.best_model_val_loss_param)
    run_metadata.evaluation_metrics = metrics

    run_metadata.save(
        run_metadata_path
    )

    return model_metadata