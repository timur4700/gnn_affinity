from datasets import register as dataset_register



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
    from models import prepare_model

    prepare_model(args.input,
                  args.model)


def run_model_train(args):
    from train import train_model

    train_model(args.input)