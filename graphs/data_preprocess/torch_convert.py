from chem.schemas import MolGraph
from utils import general

import pickle
from pathlib import Path

import torch
from torch_geometric.data import Data

from tqdm import tqdm




def data_converter(mol_graph: MolGraph) -> Data:

    data = Data()

    data.x = torch.tensor(mol_graph.x, dtype=torch.float32)
    data.edge_index = torch.tensor(mol_graph.edge_index, dtype=torch.long)
    data.pos = torch.tensor(mol_graph.pos, dtype=torch.float32)
    data.y = torch.tensor(mol_graph.y, dtype=torch.float32)

    return data



def dataset_converter(source: Path,
                      destination: Path):

    dataset: list[MolGraph] = general.unpack_pickle(source)

    with open(destination, 'wb') as f:
        for data in tqdm(dataset,
                         desc='Converting to tensors'):
            pickle.dump(data_converter(data), f)

    

    


