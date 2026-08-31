import argparse
from cli import commands


def make_parser():

    parser = argparse.ArgumentParser('gnn')


    subparsers = parser.add_subparsers(dest='command',
                                       required=True)

    
    # High-level commands
    dataset = subparsers.add_parser('dataset', help='Dataset Preprocesser SubProgram')
    graph = subparsers.add_parser('graph', help='Graph Builder SubProgram')
    model = subparsers.add_parser('model', help='Model SubProgramm')

    dataset_commands = dataset.add_subparsers(dest='command',
                                              required=True)


    # Dataset SubProgram commands
    prepare_dataset = dataset_commands.add_parser('prepare')



    prepare_dataset.add_argument('-d', '--dataset', 
                         required=True)

    prepare_dataset.add_argument('-i', '--input',
                         required=True)

    prepare_dataset.set_defaults(func=commands.run_dataset_preparation)



    # Graph SubProgram commands
    graph_commands = graph.add_subparsers(dest='command',
                                          required=True)


    graph_config = graph_commands.add_parser('config')
    graph_prepare = graph_commands.add_parser('prepare')

    graph_config.add_argument('-m', '--model',
                              required=True)

    graph_config.add_argument('-i', '--input',
                              required=True)

    graph_config.set_defaults(func=commands.run_graph_config)



    graph_prepare.add_argument('-i', '--input',
                       required=True)


    graph_prepare.add_argument('-o', '--output',
                       required=True)

    graph_prepare.set_defaults(func=commands.run_graph_preparation)


    model_commands = model.add_subparsers(dest='command',
                                          required=True)

    model_build = model_commands.add_parser('build', 
                                            help='Build registered DL model in the program')
    
    model_build.add_argument('-m', '--model', required=True,
                             help='Model name')
    
    model_build.add_argument('-i', '--input', required=True,
                             help='Path to prepared dataset')
    
    model_build.set_defaults(func=commands.run_model_prep)

    model_train = model_commands.add_parser('train', 
                                            help='Starting training model')
    
    model_train.add_argument('-i', '--input', help='The path to directory with prepared model')

    model_train.set_defaults(func=commands.run_model_train)


    model_list = model_commands.add_parser('list', help='Show available models')
    model_list.set_defaults(func=commands.get_model_list)


    return parser