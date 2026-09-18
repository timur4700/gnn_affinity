from .main import start_trainer
from .utils import find_metadata, check_metadata
from pathlib import Path

from utils import general




def train_model(
        model_directory: str,
        training_type: str
    ):

    model_directory = Path(model_directory)

    start_trainer(
        model_directory,
        training_type)