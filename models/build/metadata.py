from dataclasses import dataclass, asdict
from utils import general




@dataclass
class MetaData:
    model_name: str = ''
    id: str = None
    model_params: dict = None
    model_init_params_path: str = ''
    model_saved_params: str = None
    dataset_metadata: dict = None
    model_config_path: str = ''
    trainer_config_path: str = ''

    def save(self, path):
        general.save_json(asdict(self),
                          path)
