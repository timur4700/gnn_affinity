from dataclasses import dataclass
from typing import Any

@dataclass
class Model:
    model_name: str = ''
    model_class: Any = None
    features: dict = None
    default_params: Any = None
    custom_params: dict = None