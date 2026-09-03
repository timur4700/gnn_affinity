from dataclasses import fields, field
from typing import Sequence, Any



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



def _object_encoder(obj):

    """
    Converts Path objects into str
    
    """

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {k: _object_encoder(v) for k, v in obj.items()}

    return obj
