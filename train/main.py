from pathlib import Path
from typing import Literal

from metadata.model import ModelMetaData
from metadata.base import find_metadata, MetaDataUpdater

from train import helpers

from train.workflows import TRAINING_MODES


def start_trainer(
        model_directory: Path,
        training_type: Literal['new', 'resume']='resume'
):

    model_metadata, model_metadata_path = find_metadata(model_directory,
                                                  ModelMetaData)


    with MetaDataUpdater(model_metadata, model_metadata_path) as metadata:
        runs = helpers.inspect_runs(
            metadata
        )
        metadata.model_runs = runs

    training_func = TRAINING_MODES[training_type]


    model_metadata = training_func(
        model_metadata
    )


    with MetaDataUpdater(model_metadata, model_metadata_path) as metadata:
        runs = helpers.inspect_runs(
            metadata
        )
        metadata.model_runs = runs