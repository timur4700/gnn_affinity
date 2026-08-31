from abc import ABC, abstractmethod
from utils.schemas import LigandData, ProteinData, Features
from graphs import utils

from torch_geometric.data import Data

from typing import Any

from pathlib import Path
from graphs.configs import structure


parent_path = Path(__file__).resolve().parent


class GraphBuilder(ABC):

    def __init__(self):
        self.config = None


    def prepare_graph(self,
                      ligand_data: LigandData,
                      protein_data: ProteinData,
                      data: dict) -> Data: pass




class GeneralGraphBuilder(GraphBuilder):

    """
    Prepares standard Union Graph, ie the protein and ligand graph represented in one
    feature / adjacency matrix
    
    """

    name = 'general_graph'
    config_path = parent_path.parent / 'configs' / 'general_graph.yaml'

    def __init__(self, 
                 graph_config,
                 features: Features):

        self.graph_config = graph_config
        self.features = features
    
    def prepare_graph(self, 
                      ligand_data: LigandData, 
                      protein_data: ProteinData, 
                      data: dict[str, Any]):
  
    
        ligand_graph = utils.make_graph(ligand_data,
                                           **self.graph_config.graph,
                                           features=self.features)
            
        protein_graph = utils.make_graph(protein_data, 
                                            **self.graph_config.graph,
                                            features=self.features)
    
        graph = utils.combine_graphs(ligand_graph,
                                     protein_graph,
                                     **self.graph_config.interaction)
        
        utils.insert_data2graph(graph,
                                data)
    
        return graph
