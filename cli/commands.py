from pathlib import Path
from utils import general

from registers import datasets
from metadata.datasets import DatasetMetadata

import os


def run_dataset_preparation(args):
    module = datasets.get_dataset_module(args.dataset)
    module.prepare(args.input)




def run_graph_config(args):
    from graphs.configs.prepare import make_config

    make_config(args.input,
                args.model)




def run_graph_preparation(args):
    general.limit_threads()
    
    from graphs.data_preprocess import prepare

    dataset_meta_path = Path(args.input)
    dataset_metadata = DatasetMetadata(**general.load_json(dataset_meta_path))


    prepare.preprocess_graph_dataset(dataset_metadata,
                                     Path(args.output))



def get_model_list(args):
    from registers.models.models import MODEL_REGISTER

    print('\nAvailable Models:')
    print('-----------------')

    for k in MODEL_REGISTER.keys():
        print(f"- {k}")


def get_dataset_list(args):
    from registers.datasets import DATASET_REGISTER

    print('\nAvailable Datasets:')
    print('-----------------')

    for k in DATASET_REGISTER.keys():
        print(f"- {k}")




def run_model_prep(args):
    from models import prepare_model

    prepare_model(args.input,
                  args.model)


def run_model_train(args):
    from train import train_model

    train_model(args.input)