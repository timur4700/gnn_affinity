from datasets.mol import configs
from pathlib import Path

PARENT =Path(__file__).parents[1]


MOL_CONFIG = {
    'pdbbind':  configs.PDBbindMolConfig
    }



def load_config_path(dataset_name: str):
    return (PARENT / dataset_name / 'config.yaml').resolve()