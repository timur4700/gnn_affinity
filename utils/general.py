import yaml
import json
from pathlib import Path
from uuid import uuid4

import pickle
import os


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


def limit_native_threads(n: str='1'):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(var, n)



def mco(msg,
        choices: list[str]):

    print(msg)
    print('\n'.join([f"{i}) {choice}" for i, choice
                     in enumerate(choices)]))

    while True:
        try:
            user_choice = int(input('INPUT: '))
            if user_choice < 0:
                raise IndexError
            
            return user_choice
        except Exception:
            print('Wrong input!')