import torch
from torch.nn import MSELoss
from torch.optim import Adam, AdamW

from scipy.stats import pearsonr, spearmanr
import numpy as np

from torch_geometric.data import Data
from schemas.train import DataSetSplited

from typing import Literal, Any
import os
from pathlib import Path
import re

from typing import OrderedDict


from utils import general


def make_optimizer(optim: Literal['adam', 'adam_w']):

    if optim == 'adam':
        return Adam

    return AdamW


def make_loss_func(loss: Literal['mse']):

    if loss == 'mse':
        return MSELoss


def rp_calc(y_test: np.ndarray, y_hat: np.ndarray):
    return pearsonr(y_test, y_hat)[0]


def rs_calc(y_test: np.ndarray, y_hat: np.ndarray):
    return spearmanr(y_test, y_hat)[0]





def general_spliter(dataset: list[Data],
                    seed: int=42,
                    train_frac: float=0.9,
                    val_frac: float=0.05):


    array = np.array(dataset, dtype=object)
    total = len(dataset)


    test_frac = 1.0 - train_frac - val_frac

    if train_frac + val_frac + test_frac > 1.0:
        raise ValueError('Split fractions exceed 1')

    generator = np.random.default_rng(seed=seed)
    perm = generator.permutation(total)

    train_id, val_id, test_id = (perm[:int(total*train_frac)],
                                 perm[int(total*train_frac):int(total*train_frac + total*val_frac)],
                                 perm[int(total*train_frac + total*val_frac):])



    return DataSetSplited(train=list(array[train_id]),
                          val=list(array[val_id]),
                          test=list(array[test_id]))



def search_spliter():

    pass



def find_metadata(model_directory: Path) -> dict[str, Any] | None:
    pattern = r'metadata'

    for file in os.listdir(model_directory):
        if re.findall(pattern=pattern,
                      string=file):

            return general.load_json(model_directory/file)

    return None


def check_metadata(model_metadata: dict[str, Any]) -> dict | None:

    if model_metadata is None:
        print("Model's metadata not found")
        return None

    print(f'Metadata successfully found')

    model_name = model_metadata["model_name"]

    print(f"Training model: {model_name}")


    dataset_path = model_metadata['dataset_metadata']['dataset_path']

    if not os.path.exists(dataset_path):
        print(f"The dataset stated in metadata was not found")
        return None

    print('Dataset was found')

    train_config = model_metadata['trainer_config_path']

    if not os.path.exists(train_config):
        return None

    train_config = general.load_yaml(train_config)


    return train_config


def get_run_id(parameters_directory: Path):
    return len(os.listdir(parameters_directory))


def check_device(device) -> bool:

    if device == 'cuda':
        if torch.cuda.is_available():
            return device
        else:
            print('WARRNING: CUDA is not available on this machine')
            print('Falling back on CPU')
            device = 'cpu'

    elif device == 'mps':
        if torch.backends.mps.is_available():
            return device
        else:
            print('WARNING: MPS is not available on this machine')
            print('Falling back on CPU')
            device = 'cpu'

    return device


def save_model_run(model_metadata: dict,
                   model_params: OrderedDict,
                   model_config: dict,
                   train_config: dict,
                   metrics_test: dict,
                   train_log: str):

    parameters_directory = Path(model_metadata['model_saved_params'])


    run_path = parameters_directory / f'run_{get_run_id(parameters_directory)}'
    os.makedirs(run_path)

    torch.save(model_params, run_path / 'model.pt')
    general.save_yaml(model_config, run_path / 'model_config.yaml')
    general.save_yaml(train_config, run_path / 'train_config.yaml')
    general.save_json(metrics_test, run_path / 'metrics.json')

    with open(run_path / 'train_log.txt', 'w') as f:
        f.write(train_log)





