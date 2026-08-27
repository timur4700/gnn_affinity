from models.build.schemas import Model
from pathlib import Path
from utils import general

settings_path = Path(__file__).resolve().parent.parent / 'built_in_models' / 'all_settings'


def egnn_interaction_loader():
    from models.built_in_models import egnn_interaction
    from models.built_in_models.egnn_interaction import features

    model_name = 'egnn_interaction'
    custom_settings = settings_path / (model_name + '.yaml')


    return Model(model_name=model_name,
                  model_class=egnn_interaction.model.EgnnInteraction,
                  features=features.egnn_interaction_features(),
                  default_params=egnn_interaction.settings.ModelSettings(),
                  custom_params=general.load_yaml(custom_settings))

