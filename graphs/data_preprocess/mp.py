from utils import general
from graphs.maker.manager import GraphManager

from tqdm import tqdm
import pickle
from pathlib import Path

from multiprocessing import Pool



def init_worker(graph_manager: GraphManager):

    general.limit_native_threads()
    
    global _graph_manager
    _graph_manager = graph_manager





def main_worker(complex_dir):
        try:
            _graph_manager.set_data(complex_dir)

            result = _graph_manager.init_graph_preparation()
            return result, None

        except Exception as e:
             return None, str(complex_dir.name)



def mp_prepare(graph_manager: GraphManager,
               save_path: Path,
               n_proc: int,
               entries_paths: list[Path],
               chunk_size: int):

    """
    MP graph Preprocesser
    
    """

    failed_complexes = list()


    n_entries = len(entries_paths)


    with open(save_path, 'wb') as f:
        graphs_prepared = 0
        
        with Pool(n_proc, init_worker, (graph_manager,)) as pool:
                    
            try:
                for result, error in tqdm(pool.imap(main_worker, 
                                                entries_paths,
                                                chunksize=chunk_size), 
                                                total=n_entries):
            
                    if error is not None:
                        failed_complexes.append(error)
                        continue

                    pickle.dump(result, f)
                    graphs_prepared += 1

            except Exception as e:

                print(e)
                raise
                        
            finally:
                pool.close()
                pool.join()


                return graphs_prepared, failed_complexes