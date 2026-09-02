from datasets import register as dataset_register
from pathlib import Path
from utils import general

import os

def limit_threads():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def run_dataset_preparation(args):
    module = dataset_register.get_dataset_module(args.dataset)
    module.prepare(args.input)




def run_graph_config(args):
    from graphs.configs.prepare import make_config

    make_config(args.input,
                args.model)




def run_graph_preparation(args):
    limit_threads()
    from graphs.data_preprocess import prepare
    from datasets.metadata import DatasetMetadata


    dataset_meta_path = Path(args.input)
    dataset_metadata = DatasetMetadata(**general.load_json(dataset_meta_path))


    prepare.preprocess_graph_dataset(dataset_metadata,
                                     Path(args.output))



def get_model_list(args):
    from models.build.model_register import MODEL_REGISTER

    print('\nAvailable Models:')
    print('-----------------')

    for k in MODEL_REGISTER.keys():
        print(f"- {k}")


def run_model_prep(args):
    from models import prepare_model

    prepare_model(args.input,
                  args.model)


def run_model_train(args):
    from train import train_model

    train_model(args.input)