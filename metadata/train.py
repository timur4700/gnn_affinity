from pydantic import Field
from metadata.base import MetaData

from pathlib import Path

from typing import ClassVar, Annotated, Any, Literal


class RunMetaData(MetaData):

    metadata_name: str = 'Model Run Metadata'
    _metadata_name: ClassVar[str] = 'Model Run Metadata'

    id: str | None = None
    run_id: int | None = None
    status: Literal['not_started',
                    'training',
                    'finished', 
                    'interupted'] = 'not_started'

    epochs: int = 0

    model_config_path: Annotated[
        Path | None,
        Field(
            default=None,
            description='Model Configuration File',
            json_schema_extra={'file_required': True}
        )
    ]

    train_config_path: Annotated[
        Path | None,
        Field(
            default=None,
            description='Training Configuration File',
            json_schema_extra={'file_required': True}
        )
    ]

    model_checkpoint_path: Annotated[
        Path | None,
        Field(
            default=None,
            description='Model Checkpointer File',
            json_schema_extra={'file_required': True}
        )
    ]

    model_weights_path: Annotated[
        Path | None,
        Field(
            default=None,
            description='Model Weights File',
            json_schema_extra={'file_required': True}
        )
    ]

    train_log_path: Annotated[
        Path | None,
        Field(
            default=None,
            description='Training Log File',
            json_schema_extra={'file_required': False}
        )
    ]

    evaluation_metrics_path: Annotated[
        Path | None,
        Field(
            default=None,
            description='Evaluation Metrics File',
            json_schema_extra={'file_required': False}
        )
    ]
    evaluation_metrics: dict[str, Any] | None = None

    def make_file_paths(
            self,
            run_directory
    ) -> None:

        self.model_checkpoint_path = (
            run_directory /
            'model_checkpoint.pt'
        )

        self.model_weights_path = (
            run_directory /
            'model_weights.pt'
        )

        self.model_config_path = (
            run_directory /
            'model_config.yaml'
        )

        self.train_config_path = (
            run_directory / 
            'train_config.yaml'
        )

        self.evaluation_metrics_path = (
            run_directory /
            'metrics.json'
        )

        self.train_log_path = (
            run_directory /
            'train.log'
        )

    @classmethod
    def load_from_file(cls, metadata_path):

        return super().load_from_file(metadata_path)
