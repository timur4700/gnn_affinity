from .dataset_prep import main as main_prep
from .main import preprocess_dataset

from pathlib import Path


def prepare(input: str):

    input = Path(input)
    main_prep(input)


def make_graphs(input: str, 
                data: str, 
                output: str, 
                model: str):

    
    preprocess_dataset(Path(input), 
                       Path(data),
                       Path(output),
                       model)