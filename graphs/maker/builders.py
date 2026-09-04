from abc import ABC
from schemas.mol import LigandData, ProteinData, Features, MolGraph

from pathlib import Path

from graphs import utils
from configs.graph import ComplexGraphConfig

from dataclasses import asdict


parent_path = Path(__file__).resolve().parent


class GraphBuilder(ABC):

    def __init__(self):
        self.config = None


    def prepare_graph(self,
                      ligand_data: LigandData,
                      protein_data: ProteinData) -> MolGraph: pass



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

        self.graph_config: ComplexGraphConfig = graph_config
        self.features = features
    
    def prepare_graph(self, 
                      ligand_data: LigandData, 
                      protein_data: ProteinData):
  
    
        ligand_graph = utils.make_graph(ligand_data,
                                        **asdict(self.graph_config.ligand),
                                        features=self.features)
            
        protein_graph = utils.make_graph(protein_data, 
                                         **asdict(self.graph_config.protein),
                                         features=self.features)
    
        graph = utils.combine_graphs(ligand_graph,
                                     protein_graph,
                                     **asdict(self.graph_config.interaction))
    
        return graph