from pathlib import Path
from utils import general, options

import os
import re

from typing import Type, ClassVar, Self
from pydantic import BaseModel

from exceptions.metadata import MetaDataNotFound
from typing import get_args

from configs.base import config_validation_wrapp


def _is_path_annotation(annotation):

    return (
        annotation is Path
        or Path in get_args(annotation) 
    )


def _check_files_in_metadata(metadata_class: Type[BaseModel],
                             metadata: BaseModel):

    for k, v in metadata_class.model_fields.items():

        if not v.description:
            continue

        if not _is_path_annotation(v.annotation):
            continue
            
        file_path= getattr(metadata, k)

        if not isinstance(file_path, Path):
            continue

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{v.description} was not found")

        print(f"{v.description} found: {str(getattr(metadata, k))}")


    
class MetaData(BaseModel):
    metadata_name: str = ''
    _metadata_name: ClassVar[str] = ''

    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            data = self.model_dump(
                exclude_unset=True,
                mode='json'
            )
            
            data['metadata_name'] = self.metadata_name

            general.save_json(
                data,
                path
            )


    @classmethod
    def load_from_file(
        cls,
        metadata_path: Path
    ) -> BaseModel:

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"{cls.metadata_name} was not found")

        raw_data = general.load_json(metadata_path)
        metadata = cls.model_validate(raw_data)

        if metadata.metadata_name != cls._metadata_name:
            return

        print(f"{cls._metadata_name} was found and loaded")

        _check_files_in_metadata(cls, 
                                 metadata)

        return metadata


    @classmethod
    @config_validation_wrapp
    def load_data(
        cls, data: dict
    ) -> Self:

        """
        Loads from object data

        Parameters
        ----------
            data : dict
                Dict metadata

        Returns
        -------
            pydantic.BaseModel
        """

        return cls.model_validate(data)

    

def find_metadata(directory_path: Path,
                  metadata: Type[MetaData]) -> MetaData:

    files_in_dir = os.listdir(directory_path)
    metadata_name = metadata._metadata_name

    for file in files_in_dir:

        if re.search(r'metadata', file):

            metadata_path = directory_path / file
            loaded_metadata = metadata.load_from_file(metadata_path)

            if loaded_metadata is None:
                continue

            return loaded_metadata

    raise MetaDataNotFound(
        metadata_name, 
        directory_path
    )