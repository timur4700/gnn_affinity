from abc import ABC
from schemas.mol import LigandData, ProteinData, Features, MolGraph

from pathlib import Path

from graphs import utils
from configs.graph import ComplexGraphConfig

from dataclasses import asdict


parent_path = Path(__file__).resolve().parent


class GraphBuilder(ABC):

    """
    Abstract base class for constructing molecular graph representations.

    Defines the interface for graph preparation from ligand and protein
    molecular data. Concrete implementations specify how molecular
    components and their interactions are represented.

    Attributes
    ----------
    config : Any or None
        Configuration associated with the graph builder. Initialized
        to None and intended to be assigned by concrete implementations.
    """


    def __init__(self):
        self.config = None


    def prepare_graph(
            self,
            ligand_data: LigandData,
            protein_data: ProteinData
    ) -> MolGraph: 

        """
        Initialize the graph builder.

        Sets the configuration attribute to None.
        """
        
        pass

    

class GeneralGraphBuilder(GraphBuilder):

    """
    Construct a unified graph representation of a protein-ligand complex.

    Represents the ligand, protein, and their interactions within a
    single graph structure, combining their node and edge features.

    Attributes
    ----------
    name : str
        Identifier of the graph builder ('general_graph').
    config_path : pathlib.Path
        Path to the default YAML configuration file.
    graph_config : ComplexGraphConfig
        Configuration defining graph construction parameters for the
        ligand, protein, and intermolecular interactions.
    features : Features
        Feature configuration used to generate molecular node and
        edge representations.
    """

    name = 'general_graph'
    config_path = parent_path.parent / 'configs' / 'general_graph.yaml'

    def __init__(
            self, 
            graph_config,
            features: Features
        ):

        super().__init__()


        """
        Initialize the general graph builder.

        Parameters
        ----------
        graph_config : ComplexGraphConfig
            Configuration specifying graph construction settings for
            ligand, protein, and interaction graphs.
        features : Features
            Feature configuration used to generate molecular graph
            representations.
        """


        self.graph_config: ComplexGraphConfig = graph_config
        self.features = features
    
    def prepare_graph(
            self, 
            ligand_data: LigandData, 
            protein_data: ProteinData
        ) -> MolGraph:


        """
        Construct a unified protein-ligand graph.

        Generates separate graph representations for the ligand and
        protein, then combines them into a single graph according to
        the configured intermolecular interaction settings.

        Parameters
        ----------
        ligand_data : LigandData
            Processed ligand molecular data used to construct the
            ligand graph.
        protein_data : ProteinData
            Processed protein molecular data used to construct the
            protein graph.

        Returns
        -------
        MolGraph
            Unified graph containing ligand and protein representations
            together with their intermolecular interaction edges.

        Notes
        -----
        Graph construction involves three steps:

        1. Construct the ligand graph using the ligand configuration.
        2. Construct the protein graph using the protein configuration.
        3. Combine both graphs using the interaction configuration.

        The same feature configuration is used for both molecular
        components.
        """
  
    
        ligand_graph = utils.make_graph(
            ligand_data,
            **self.graph_config.ligand.model_dump(),
            features=self.features
        )
            
        protein_graph = utils.make_graph(
            protein_data, 
            **self.graph_config.protein.model_dump(),
            features=self.features
        )
    
        graph = utils.combine_graphs(
            ligand_graph,
            protein_graph,
            **self.graph_config.interaction.model_dump()
        )
    
        return graph