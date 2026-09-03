from schemas.mol import MolGraph
from utils import general

from typing import Any

import pickle
from pathlib import Path

import torch
from torch_geometric.data import Data

from tqdm import tqdm



def insert_data2graph(graph: MolGraph,
                      data: dict[str, Any],
                      y_dtype: torch.dtype=torch.float32):

    """
    The function that unserts data to torch_geometric.data.Data object
    **Note!** The function expect, that data includes ``y`` key, ie target variable
    
    """

    if 'y' not in data:
        print('The target variable was not provided for the graph')

    for k, v in data.items():

        if k == 'y':
            setattr(graph, k, torch.tensor(v, dtype=y_dtype))
            continue

        setattr(graph, k, v)



def data_converter(mol_graph: MolGraph) -> Data:

    data = Data()

    data.x = torch.tensor(mol_graph.x, dtype=torch.float32)
    data.mol_id = torch.tensor(mol_graph.mol_id, dtype=torch.long)
    data.pos = torch.tensor(mol_graph.pos, dtype=torch.float32)
    data.edge_index = torch.tensor(mol_graph.edge_index, dtype=torch.long)
    data.edge_type = torch.tensor(mol_graph.edge_type, dtype=torch.long)

    insert_data2graph(data, mol_graph.data)

    return data



def dataset_converter(source: Path,
                      destination: Path,
                      data_checker: Any=None):

    dataset: list[MolGraph] = general.unpack_pickle(source)

    with open(destination, 'wb') as f:
        for data in tqdm(dataset, desc='Converting to tensors'):

            if data_checker is not None:
                if data_checker(data):

                    pickle.dump(data_converter(data), f)