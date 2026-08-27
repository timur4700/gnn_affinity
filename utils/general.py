import yaml
import json
from pathlib import Path
from uuid import uuid4

import pickle




def load_yaml(path: str | Path) -> dict:
    """
    Loads .yaml configuration files
    """

    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(object: dict, 
              path: str | Path) -> None:
    """
    Saves dict object as .yaml configuration file
    
    """

    with open(path, 'w') as f:
        yaml.safe_dump(object, f)



def save_json(object: dict,
              path: Path) -> None:

    with open(path, 'w') as f:
        json.dump(object, f, indent=4)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)



def make_unique_id(length=6):
    return uuid4().hex[:length+1]



def unpack_pickle(path: Path):
    data = []

    with open(path, 'rb') as f:

        while True:
            try:
                data.append(pickle.load(f))
    
            except EOFError:
                break


    return data