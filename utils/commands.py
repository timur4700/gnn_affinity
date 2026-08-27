from datasets import register as dataset_register
from models import prepare_model


def run_dataset_preparation(args):
    module = dataset_register.get_dataset_module(args.dataset)
    module.prepare(args.input)


def run_graph_preparation(args):
    module = dataset_register.get_dataset_module(args.dataset)
    module.make_graphs(args.input,
                       args.target,
                       args.output,
                       args.model)


def run_model_prep(args):

    prepare_model(args.input,
                  args.model)