from .dataset_prep import main as main_prep


from pathlib import Path


def prepare(input: str):

    input = Path(input)
    main_prep(input)




def get_config():
    return Path(__file__).resolve().parent / 'config.yaml'
