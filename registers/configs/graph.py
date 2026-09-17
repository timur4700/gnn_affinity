from configs import graph
from pathlib import Path

PARENT = Path(__file__).parents[2]

GRAPH_CONFIG = {
    'general_graph': graph.ComplexGraphConfig 
}


def load_config_path(graph_name: str):
    return (
        PARENT / 
        'datasets' / 
        'configs' / 
        f'{graph_name}.yaml'
    ).resolve()