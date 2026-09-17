from configs.preprocessing import ConfigData
from exceptions.model import WrongGraphConfig

def check_interaction(graph) -> bool:
    edge_type = graph.edge_type

    if not 2 in edge_type:
        return False

    return True



def graph_config_validator(
        config: ConfigData
    ) -> ConfigData:

    if not config.graph_config.interaction.add_interaction_edges:
        raise WrongGraphConfig(
            (
                'egnn_interaction model has to include explicit interation edges\n'
                'change add_interaction_edges on -> true'

            )
        )

    return config