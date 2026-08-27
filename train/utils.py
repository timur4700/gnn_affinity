from torch.nn import MSELoss
from torch.optim import Adam, AdamW

from scipy.stats import pearsonr, spearmanr
import numpy as np

from torch_geometric.data import Data
from utils.schemas import DataSetSplited

from typing import Literal

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


    test_frac = train_frac - val_frac

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



