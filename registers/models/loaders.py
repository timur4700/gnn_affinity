from models.build.schemas import Model
from pathlib import Path
from utils import general

from models.build.associated_graphs import MODEL2GRAPH
from graphs import register


settings_path = Path(__file__).resolve().parent.parent / 'built_in_models' / 'all_settings'



def bind_graphs2model(model_name):

    graph_names = MODEL2GRAPH.get(model_name)

    if not graph_names:
        raise ValueError(f'The associated graphs with the model {model_name}\
                         were not found. Closing program')

    return [register.GRAPH_TYPES.get(graph) for graph
            in graph_names]

    

def egnn_interaction_loader():
    from models.built_in_models import egnn_interaction
    from models.built_in_models.egnn_interaction import features, settings, graph_anayzer

    model_name = 'egnn_interaction'
    custom_settings = settings_path / (model_name + '.yaml')


    return Model(model_name=model_name,
                 model_class=egnn_interaction.get_model_class,
                 graph_builder=bind_graphs2model(model_name),
                 graph_analyzer=graph_anayzer.check_interaction,
                 features=features.egnn_interaction_features(),
                 default_params=settings.ModelSettings(),
                 custom_params=general.load_yaml(custom_settings))
