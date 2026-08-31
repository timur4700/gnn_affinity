import os
from pathlib import Path
from utils import general

from datasets.pdbbind import metadata
from datasets import metadata

from graphs.metadata import GraphDatasetMeta, make_graph_metadata
from graphs.maker.manager import GraphManager
from graphs.maker.builders import GeneralGraphBuilder
from graphs.configs.structure import GraphConfig
from graphs import metadata as graph_metadata
from graphs.data_preprocess import helpers, mp



def make_saving_paths(destination_path: Path,
                      dataset_id: int) -> graph_metadata.SavingPaths:

    return graph_metadata.SavingPaths.make(destination_path,
                                           dataset_id)


def make_graph_mol_config(dataset_metadata: metadata.DatasetMetadata):

    graph_config = helpers.load_config(dataset_metadata.graph_config_path,
                                           'GraphConfig',
                                           GraphConfig) 
    
    mol_config = helpers.load_mol_config(dataset_metadata.mol_config_path,
                                         dataset_metadata.name)
    
    features = helpers.make_features(dataset_metadata.model)

    return graph_metadata.ConfigData(graph_config=graph_config,
                                     mol_config=mol_config,
                                     features=features)


def make_entries_path_data(dataset_metadata: metadata.DatasetMetadata):

    entries_dir = Path(dataset_metadata.entries_path)

    entries_paths = [entries_dir / pdb for pdb 
                        in os.listdir(entries_dir)]

    entries_data = helpers.load_target_data(dataset_metadata.target_path)
    

    return graph_metadata.EntriedData(entries_paths,
                                      entries_data)



def calc_chunck_n_proc(entries_paths):
    n_proc = os.cpu_count()
    chunk_size = helpers.chunk_size_calc(entries_paths)

    return {'n_proc': n_proc, 'chunk_size': chunk_size}



def collect_results(results,
                    preproc_data: graph_metadata.PreprocessingData,
                    graph_dataset_metadata: GraphDatasetMeta):

    graphs_prepared, failed = results
    error_log = helpers.make_error_log(failed)

    with open(preproc_data.saving_paths.error_log, 'w') as f:
        f.write(error_log)

    graph_dataset_metadata.graph_num = graphs_prepared
    graph_dataset_metadata.status = 'completed'


def make_preproc_data(dataset_metadata,
                      destination_path):

    dataset_id = general.make_unique_id()
    entries = make_entries_path_data(dataset_metadata)

    preprocess_data = graph_metadata.PreprocessingData(
        id=dataset_id,
        saving_paths= make_saving_paths(destination_path, dataset_id),
        configs=make_graph_mol_config(dataset_metadata),
        entries=entries,
        mp_config=graph_metadata.MpConfig(
        **calc_chunck_n_proc(entries.paths)
            )
        )

    return preprocess_data


def make_graph_manager(dataset_metadata: metadata.DatasetMetadata,
                       preproc_data: graph_metadata.PreprocessingData):

    path_builder = helpers.get_path_builder(dataset_metadata.name)

    graph_builder=GeneralGraphBuilder(preproc_data.configs.graph_config, 
                                      preproc_data.configs.features)

    return GraphManager(
                path_builder=path_builder,
                graph_builder=graph_builder,
                target_data=preproc_data.entries.target,
                mol_config=preproc_data.configs.mol_config,
                sanitize=False)


def make_graph_dataset_meta(preprocess_data: graph_metadata.PreprocessingData,
                            dataset_metadata: metadata.DatasetMetadata):

    graph_config_metadata = make_graph_metadata(preprocess_data.configs.features, 
                                              preprocess_data.configs.graph_config)

    return GraphDatasetMeta(model=dataset_metadata.model,
                            id=preprocess_data.id,
                            graph_config_metadata=graph_config_metadata,
                            dataset_path=preprocess_data.saving_paths.graph_dataset)



def make_postrocess_data(preprocess_data: graph_metadata.PreprocessingData):

    return make_graph_metadata(preprocess_data.configs.features,
                               preprocess_data.configs.graph_config)



def preprocess_graph_dataset(dataset_metadata: metadata.DatasetMetadata,
                             destination_path: Path):


    preprocess_data = make_preproc_data(dataset_metadata,
                                        destination_path)

    preprocess_data.saving_paths.make_dir()

    graph_manager = make_graph_manager(dataset_metadata,
                                       preprocess_data)

    graph_dataset_metadata = make_graph_dataset_meta(preprocess_data,
                                                     dataset_metadata)



    results = mp.mp_prepare(
        graph_manager,
        preprocess_data.saving_paths.graph_dataset,
        preprocess_data.mp_config.n_proc,
        preprocess_data.entries.paths,
        preprocess_data.mp_config.chunk_size
        )

    collect_results(results,
                    preprocess_data,
                    graph_dataset_metadata)

    graph_dataset_metadata.save(preprocess_data.saving_paths.metadata)
    
    print(f"PDBbind Graph Dataset was successfully saved in\
           {str(preprocess_data.saving_paths.graph_dataset_dir)}")

    


    