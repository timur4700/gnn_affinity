from dataclasses import dataclass, asdict, fields, field

from pathlib import Path
from utils import general, options

import os
import re

from typing import Type
from pydantic import BaseModel

@dataclass
class MetaData:

    metadata_name: str = ''

    def __post_init__(self):
        options.convert2object(self, Path)


    def save(self, path: Path):
        general.save_json(options._object_encoder(asdict(self)),
                          path)


    def _validate(self):
        options.check_files_in_metadata(self)


    @classmethod
    def load(cls, metadata_path: Path):
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f'{cls.metadata_name} was not found')

        raw_data = general.load_json(metadata_path)
        metadata = cls(**raw_data)

        if metadata.metadata_name != cls.metadata_name:
            return
        
        print(f"{cls.metadata_name} found")
        
        metadata._validate()

        return metadata



def find_metadata(directory_path: Path,
                  metadata: MetaData) -> MetaData:

    files_in_dir = os.listdir(directory_path)
    metadata_name = metadata.metadata_name

    for file in files_in_dir:
        if re.search(r'metadata', file):

            metadata_path = directory_path / file
            loaded_metadata = metadata.load(metadata_path)

            if loaded_metadata is None:
                continue

            return loaded_metadata

    raise FileNotFoundError(
        f"{metadata_name} was not found in directory: {directory_path}"
        )




def _check_files_in_metadata(metadata_class: Type[BaseModel],
                             metadata: BaseModel):

    for k, v in metadata_class.model_fields.items():

        if (v.annotation == Path and
            v.description):

            if not os.path.exists(getattr(metadata,
                                          k)):
                raise FileNotFoundError(f"{v.description} was not found")

            print(f"{v.description} found: {str(getattr(metadata, k))}")


    
class MetaDataPy(BaseModel):
    metadata_name: str = ''


    def save(self, path: Path) -> None:
        general.save_json(self.model_dump(),
                          path)


    @classmethod
    def load(
        cls,
        metadata_path: Path
    ) -> BaseModel:

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"{cls.metadata_name} was not found")

        raw_data = general.load_json(metadata_path)
        metadata = cls.model_validate(raw_data)

        if metadata.metadata_name != cls.metadata_name:
            return

        print(f"{cls.metadata_name} was found and loaded")

        _check_files_in_metadata(cls, 
                                 metadata)

        return metadata