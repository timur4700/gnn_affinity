from pathlib import Path



PARENT = Path(__file__).parent


def load_model_config_path(model_name: str):
    return PARENT / model_name 