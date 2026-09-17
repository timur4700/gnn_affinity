from metadata.model import ModelMetaData
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


def status_wrapper(
        func,
        run_metadata, 
        run_metadata_path,
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
            save_and_update_status(
            run_metadata,
            'finished',
            run_metadata_path
            )
            return results

        except Exception:
            save_and_update_status(
                run_metadata,
                'interupted',
                run_metadata_path
            )



    return wrapped
            




