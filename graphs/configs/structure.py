from dataclasses import dataclass
from typing import Any




@dataclass
class GraphConfig:
    graph: dict[str, Any]
    interaction: dict[str, Any]


    @classmethod
    def load_data(cls, data: dict):
        return cls(
            graph=data['Graph'],
            interaction=data['Interaction']
        )