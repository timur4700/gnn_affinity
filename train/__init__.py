from .main import start_trainer
from .utils import find_metadata, check_metadata
from pathlib import Path

from utils import general




def train_model(model_directory: str):

    model_directory = Path(model_directory)


    #if train_config is None:
    #    print('The trainer configuration file was not found in the model directory')
    #    print('Using default configuration')
    #    train_config = general.load_yaml('settings.yaml')


    start_trainer(Path(model_directory))