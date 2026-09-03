import importlib



DATASET_REGISTER = {
    'pdbbind': 'datasets.pdbbind'
}



def get_dataset_module(name: str):
    if name not in DATASET_REGISTER:
        raise ValueError()

    return importlib.import_module(DATASET_REGISTER[name])