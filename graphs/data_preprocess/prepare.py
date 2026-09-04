import os
from pathlib import Path
import shutil

from utils import general


from metadata.datasets import DatasetMetadata
from metadata.graphs import GraphDatasetMeta

from configs.utils import make_graph_metadata
from configs import preprocessing

from graphs.maker.manager import GraphManager
from graphs.maker.builders import GeneralGraphBuilder


from configs.graph import ComplexGraphConfig

from graphs.data_preprocess import helpers, mp

from tempfile import TemporaryDirectory


def make_saving_paths(destination_path: Path,
                      dataset_id: int) -> preprocessing.SavingPaths:

    return preprocessing.SavingPaths.make(destination_path,
                                           dataset_id)




def make_graph_mol_config(dataset_metadata: DatasetMetadata):

    graph_config = helpers.load_config(dataset_metadata.graph_config_path,
                                           'GraphConfig',
                                           ComplexGraphConfig) 
    
    mol_config = helpers.load_mol_config(dataset_metadata.mol_config_path,
                                         dataset_metadata.name)
    
    features = helpers.make_features(dataset_metadata.model)

    return preprocessing.ConfigData(graph_config=graph_config,
                                     mol_config=mol_config,
                                     features=features)


def make_entries_path_data(dataset_metadata: DatasetMetadata):

    entries_dir = Path(dataset_metadata.entries_path)

    entries_paths = [entries_dir / pdb for pdb 
                        in os.listdir(entries_dir)]

    entries_data = helpers.load_target_data(dataset_metadata.target_path)
    

    return preprocessing.EntriedData(entries_paths,
                                      entries_data)



def calc_chunck_n_proc(entries_paths: list[Path], 
                       n_cpu: int=1):
    
    chunk_size = helpers.chunk_size_calc(entries_paths,
                                         n_cpu)
    
    if n_cpu == 1:
        chunk_size = 1

    return {'n_cpu': n_cpu, 'chunk_size': chunk_size}



def collect_results(results,
                    preproc_data: preprocessing.PreprocessingData,
                    graph_dataset_metadata: GraphDatasetMeta):

    graphs_prepared, failed = results
    error_log = helpers.make_error_log(failed)

    with open(preproc_data.saving_paths.error_log, 'w') as f:
        f.write(error_log)

    graph_dataset_metadata.graph_num = graphs_prepared
    graph_dataset_metadata.status = 'completed'


def make_preproc_data(dataset_metadata,
                      destination_path,
                      n_cpu: int=1):

    dataset_id = general.make_unique_id()
    entries = make_entries_path_data(dataset_metadata)

    preprocess_data = preprocessing.PreprocessingData(
        id=dataset_id,
        model=helpers.get_model(dataset_metadata.model),
        saving_paths= make_saving_paths(destination_path, dataset_id),
        configs=make_graph_mol_config(dataset_metadata),
        entries=entries,
        mp_config=preprocessing.MpConfig(
        **calc_chunck_n_proc(entries.paths,
                             n_cpu)
            )
        )

    return preprocess_data


def make_graph_manager(dataset_metadata: DatasetMetadata,
                       preproc_data: preprocessing.PreprocessingData):

    path_builder = helpers.get_path_builder(dataset_metadata,
                                            preproc_data)

    graph_builder = GeneralGraphBuilder(preproc_data.configs.graph_config, 
                                      preproc_data.configs.features)

    return GraphManager(
                path_builder=path_builder,
                graph_builder=graph_builder,
                target_data=preproc_data.entries.target,
                mol_config=preproc_data.configs.mol_config,
                sanitize=False)


def make_graph_dataset_meta(preprocess_data: preprocessing.PreprocessingData,
                            dataset_metadata: DatasetMetadata):

    graph_config_metadata = make_graph_metadata(preprocess_data.configs.features, 
                                                preprocess_data.configs.graph_config)

    return GraphDatasetMeta(name=dataset_metadata.name,
                            model=dataset_metadata.model,
                            id=preprocess_data.id,
                            graph_config_metadata=graph_config_metadata,
                            dataset_path=preprocess_data.saving_paths.graph_dataset,
                            )



def make_postrocess_data(preprocess_data: preprocessing.PreprocessingData) -> GraphDatasetMeta:

    return make_graph_metadata(preprocess_data.configs.features,
                               preprocess_data.configs.graph_config)


def start_convert(tmp_data: Path,
                  graph_dataset_dest: Path,
                  model) -> None:
    
    from graphs.data_preprocess import torch_convert

    torch_convert.dataset_converter(tmp_data,
                                    graph_dataset_dest,
                                    model.graph_analyzer)



def preprocess_graph_dataset(dataset_metadata: DatasetMetadata,
                             destination_path: Path,
                             n_cpu: int=1):


    preprocess_data = make_preproc_data(dataset_metadata,
                                        destination_path,
                                        n_cpu=n_cpu)

    preprocess_data.saving_paths.make_dir()

    graph_manager = make_graph_manager(dataset_metadata,
                                       preprocess_data)

    graph_dataset_metadata = make_graph_dataset_meta(preprocess_data,
                                                     dataset_metadata)

    print(f"N_CPU: {preprocess_data.mp_config.n_cpu}")
    print(f"CHUNK: {preprocess_data.mp_config.chunk_size}")



    with TemporaryDirectory(dir=preprocess_data.saving_paths.graph_dataset_dir) as tmp:
        tmp = Path(tmp)
        tmp_data_path = tmp / 'preprocessed_data.pkl'

        results = mp.mp_prepare(
            graph_manager,
            tmp_data_path,
            preprocess_data.mp_config.n_cpu,
            preprocess_data.entries.paths,
            preprocess_data.mp_config.chunk_size
            )

        print('Tensor convertation will be running on 1 CPU')


        try:

            start_convert(tmp_data_path,
                          preprocess_data.saving_paths.graph_dataset,
                          preprocess_data.model)

        except ValueError as e:
            print(e)
            shutil.rmtree(preprocess_data.saving_paths.graph_dataset_dir)
            print(f'Directory {preprocess_data.saving_paths.graph_dataset_dir} was deleted')

            return
            



    collect_results(results,
                    preprocess_data,
                    graph_dataset_metadata)

    
    
    graph_dataset_metadata.save(preprocess_data.saving_paths.metadata)
    
    print(f"PDBbind Graph Dataset was successfully saved in\
           {str(preprocess_data.saving_paths.graph_dataset_dir.resolve())}")