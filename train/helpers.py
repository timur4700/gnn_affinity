from metadata.base import MetaDataUpdater
from metadata.model import ModelMetaData, RunPaths
from metadata.train import RunMetaData

from train.utils import get_run_id

from pathlib import Path
from typing import Literal

import os
import shutil

from functools import wraps

def make_run_directory(
        parent_parameters_path: Path
) -> tuple[int, Path]:

    run_id = get_run_id(
        parent_parameters_path
    )

    run_directory = (
        parent_parameters_path /
        f'run_{run_id}'
    )

    os.makedirs(run_directory)

    return (run_id, run_directory)


def make_run_metadata(
        model_metadata: ModelMetaData,
        run_id, 
        run_directory
) -> RunMetaData:

    
    model_id = model_metadata.id

    run_metadata = RunMetaData(
        id=model_id,
        run_id=run_id
    )

    run_metadata.make_file_paths(run_directory)

    return run_metadata


def transfer_configs2run_directory(
        model_metadata: ModelMetaData,
        run_metadata: RunMetaData
) -> None:

    shutil.copy2(
        model_metadata.model_config_path,
        run_metadata.model_config_path
    )

    shutil.copy2(
        model_metadata.trainer_config_path,
        run_metadata.train_config_path
    )


def save_and_update_status(
        run_metadata: RunMetaData,
        status: Literal[
            'not_started',
            'training',
            'finished', 
            'interupted'
        ],
        run_metadata_path: Path
) -> RunMetaData:

    run_metadata.status = status
    run_metadata.save(run_metadata_path)

    return run_metadata


def inspect_runs(
        model_metadata: ModelMetaData
) -> list[RunPaths]:

    runs = []

    for run in os.listdir(
        model_metadata.model_saved_params
    ):
        run_path = model_metadata.model_saved_params / run

        for file in os.listdir(run_path):
            if file == 'metadata.json':
                runs.append(
                    RunPaths(
                        name=run,
                        path=run_path/file
                    )
                )

                break

    return runs
        

def add_run2metadata(
        run_id: id,
        run_metadata_path: Path,
        model_metadata: ModelMetaData
) -> ModelMetaData:

    new_run = RunPaths(
        name=f"run_{run_id}",
        path=run_metadata_path
    )

    model_metadata.model_runs = [
        *model_metadata.model_runs, new_run
    ]

    return model_metadata


def status_wrapper(
        func,
        trainer,
        run_metadata: RunMetaData, 
        run_metadata_path: Path,
    ):

    @wraps(func)
    def wrapped(*args, **kwargs):
        
        save_and_update_status(
            run_metadata,
            'training',
            run_metadata_path
        )

        try:
            results = func(*args, **kwargs)
            run_metadata.epochs += trainer.cur_epoch

            save_and_update_status(
            run_metadata,
            'finished',
            run_metadata_path
            )

            return results

        except KeyboardInterrupt:
            run_metadata.epochs += trainer.cur_epoch

            save_and_update_status(
                            run_metadata,
                            'interupted',
                            run_metadata_path
            )

            raise

        except Exception:
            run_metadata.epochs += trainer.cur_epoch

            save_and_update_status(
                run_metadata,
                'interupted',
                run_metadata_path
            )

            raise
        
    return wrapped