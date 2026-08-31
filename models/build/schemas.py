from dataclasses import dataclass
from typing import Any

from graphs.maker.builders import GraphBuilder

@dataclass
class Model:
    model_name: str = ''
    model_class: Any = None
    graph_builder: list[GraphBuilder] = None
    features: dict = None
    default_params: Any = None
    custom_params: dict = None