from dataclasses import fields, field
from typing import Sequence, Any

from pathlib import Path
import os



def option_checker(obj):
    for field in fields(obj):
        if 'options' in field.metadata:
            options = field.metadata['options']
            option_name = field.name

            value = getattr(obj, option_name)

            if value not in options:
                raise ValueError(f'Unrecognized option <{value}> in <{option_name}> option')



def make_option_field(default_value: Any,
                      options: Sequence[Any]):
    
    return field(default=default_value,
                 metadata={'options': set(options)}) 


def make_path_field(name: str, 
                    required: bool=True):

    return field(default=None,
                 metadata={'name': name,
                           'required': required})



def _object_encoder(obj):

    """
    Converts Path objects into str
    
    """

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {k: _object_encoder(v) for k, v in obj.items()}

    return obj



def check_files_in_metadata(obj: Any):

    for field in fields(obj):
        attr_name = field.name
        metadata = field.metadata

        value = getattr(obj, attr_name)

        if field.type == Path:
            path_name = metadata.get('name', attr_name)
            if not os.path.exists(value):
                raise FileNotFoundError(f"{path_name} was not found")

            print(f"{path_name} found: {value}")



def convert2object(obj, obj_type):

    for field in fields(obj):

        if field.type == obj_type:
            value = getattr(obj, field.name)

            if value is not None:
                setattr(obj, field.name, obj_type(value))