from dataclasses import dataclass
from typing import Any
from utils import options


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



@dataclass
class GeneralGraphMolConfig:
    undirected:bool = options.make_option_field(True, [True, False])
    self_loop:bool = options.make_option_field(False, [True, False])
    graph_type:str = options.make_option_field('2d', ['2d', '3d'])
    intra_cutoff:float = 5.0

    def __post_init__(self):
       options.option_checker(self)



@dataclass
class GeneralGraphInterConfig:
   add_interaction_edges:bool = options.make_option_field(True, [True, False])
   edge_type:bool = options.make_option_field(True, [True, False])
   intra_cutoff: float = 5.0

   def __post_init__(self):
       options.option_checker(self)