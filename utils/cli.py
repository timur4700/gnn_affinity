import argparse
from utils import commands


def make_parser():

    parser = argparse.ArgumentParser('gnn')


    subparsers = parser.add_subparsers(dest='command',
                                       required=True)

    dataset = subparsers.add_parser('dataset', help='Dataset Preprocesser SubProgram')
    model = subparsers.add_parser('model', help='Model SubProgramm')

    dataset_commands = dataset.add_subparsers(dest='command',
                                              required=True)

    prepare = dataset_commands.add_parser('prepare')
    graph = dataset_commands.add_parser('graph')

    prepare.add_argument('-d', '--dataset', 
                         required=True)

    prepare.add_argument('-i', '--input',
                         required=True)

    prepare.set_defaults(func=commands.run_dataset_preparation)


    graph.add_argument('-d', '--dataset',
                       required=True)

    graph.add_argument('-m', '--model',
                       default=None,
                       required=False,
                       help='The name of the model')

    graph.add_argument('-i', '--input',
                       required=True)

    graph.add_argument('-t', '--target',
                       required=True,
                       help='The path where target variable is stored [.csv]')

    graph.add_argument('-o', '--output',
                       required=True)

    graph.set_defaults(func=commands.run_graph_preparation)


    model_commands = model.add_subparsers(dest='command',
                                          required=True)

    model_build = model_commands.add_parser('build')
    model_build.add_argument('-m', '--model', required=True)
    model_build.add_argument('-i', '--input', required=True)

    model_build.set_defaults(func=commands.run_model_prep)



    return parser