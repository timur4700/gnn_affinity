from dataclasses import dataclass
from typing import Any




@dataclass
class ComplexGraphConfig:
    ligand: dict[str, Any]
    protein: dict[str, Any]
    interaction: dict[str, Any]

    @classmethod
    def load_data(cls, data: dict):
        return cls(
            ligand=data['Graph']['Ligand'],
            protein=data['Graph']['Protein'],
            interaction=data['Interaction']
        )