from schemas.mol import Features
from graphs.utils import add_feature_padding

from configs.graph import ComplexGraphConfig
from metadata.graphs import GraphMetadata




def calculate_node_dim(features: Features):
    node_dim = min(len(features.ligand_features),
                  len(features.protein_features)) + add_feature_padding(features.ligand_features,
                                                                            features.protein_features)

    return node_dim




def make_graph_metadata(features: Features,
                        graph_config: ComplexGraphConfig) -> GraphMetadata:


    node_dim = min(len(features.ligand_features),
                   len(features.protein_features)) + add_feature_padding(features.ligand_features,
                                                                        features.protein_features)

    

    return GraphMetadata(node_dim=node_dim,
                         ligand_features=list(features.ligand_features.keys()),
                         protein_features=list(features.protein_features.keys()),
                         graph_config=graph_config)