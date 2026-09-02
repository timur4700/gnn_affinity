import numpy as np
from typing import Any
from utils.schemas import DataSetSplited




def pdbbind_spliter(dataset: list[Any],
                    seed: int=42,
                    train_frac: float=0.9):


    train_dataset_names = ['refined-set',
                           'general']

    train_dataset = [data for data in dataset if data.dataset in train_dataset_names]
    test = [data for data in dataset if data.dataset == 'coreset']

    total = len(train_dataset)

    generator = np.random.default_rng(seed=seed)
    perm = generator.permutation(total)

    train_id = perm[:int(total*train_frac)]
    val_id = perm[int(total * train_frac):]

    train = []
    val = []

    for i, data in enumerate(train_dataset):
            if i in train_id:
                    train.append(data)
                    continue

            val.append(data)

    return DataSetSplited(train=train,
                          val=val,
                          test=test)

    


