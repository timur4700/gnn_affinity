from configs import mol
from pathlib import Path

PARENT = Path(__file__).parents[2]


MOL_CONFIG = {
    'pdbbind':  mol.PDBbindMolConfig
    }



def load_config_path(dataset_name: str):
    return (PARENT / 'datasets' / 'configs' / f'{dataset_name}.yaml').resolve()