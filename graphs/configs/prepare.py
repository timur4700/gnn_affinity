from utils import general
from utils.db_call import ModelDB
from pathlib import Path
import shutil

from typing import Tuple

from registers.configs.mol import load_config_path
from metadata.datasets import DatasetMetadata


def load_dataset_metadata(dataset_directory: Path):

    try:
        metadata_raw = general.load_json(dataset_directory / 'metadata.json')
        return DatasetMetadata(**metadata_raw)

    except FileNotFoundError as e:
        raise e("Dataset's metadata was not found. Prepare dataset with command <gnn-affinity prepare>")



def user_graph_choice(model):
    graphs = model().graph_builder

    graph_names = [graph.name for graph
                    in graphs]

    choice_id = general.mco('Choose Graph Type:',
                            choices=graph_names)

    return graphs[choice_id]




def copy_configs(paths: list[Tuple[Path]]):

    for path in paths:
        source, destination = path
        shutil.copy(source,
                    destination)
    


def update_metadata(metadata: DatasetMetadata,
                    model_name: str,
                    graph_name: str,
                    graph_config_path: Path,
                    mol_config_path: Path):

    
    metadata.model = model_name
    metadata.graph = graph_name
    metadata.graph_config_path = graph_config_path
    metadata.mol_config_path = mol_config_path



def make_config(dataset_directory: str,
                model_name: str):

    dataset_directory = Path(dataset_directory)

    metadata = load_dataset_metadata(dataset_directory)

    model_db = ModelDB()
    model = model_db(model_name)

    if model is None:
        raise KeyError('The model was not found in the register. Closing program')

    graph = user_graph_choice(model)
    graph_name = graph.name
    graph_config_source = graph.config_path
    graph_config_dest = dataset_directory / 'graph_config.yaml'

    mol_config_source = load_config_path(metadata.name)
    mol_config_dest = dataset_directory / 'mol_config.yaml'

    copy_configs([(graph_config_source, graph_config_dest),
                  (mol_config_source, mol_config_dest)])
    

    update_metadata(metadata,
                    model_name,
                    graph_name,
                    graph_config_dest,
                    mol_config_dest)


    metadata.save(dataset_directory/'metadata.json')

    

    print(f'Graph Configuration can be found in {str(graph_config_dest.resolve())}')