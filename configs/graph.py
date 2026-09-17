from dataclasses import dataclass
from typing import Any, Literal, Annotated, Self
from utils import options

from pydantic import BaseModel, Field
from configs.base import config_validation_wrapp



class GraphConfig(BaseModel):
    pass



# General Graph Config
class GeneralGraphMolConfig(GraphConfig):
    undirected: Literal[True, False]
    self_loop: Literal[True, False]
    graph_type: Literal['2d', '3d']
    intra_cutoff: Annotated[float | int, Field(ge=0.0)]


class GeneralGraphInterConfig(GraphConfig):
    add_interaction_edges: Literal[True, False]
    edge_type: Literal[True, False]
    inter_cutoff: Annotated[float | int, Field(ge=0.0)] = 5.0



class ComplexGraphConfig(GraphConfig):
    ligand: GeneralGraphMolConfig
    protein: GeneralGraphMolConfig
    interaction: GeneralGraphInterConfig

    @classmethod
    @config_validation_wrapp
    def load_data(cls, data: dict[str, dict]) -> Self:

        return cls(
            ligand=GeneralGraphMolConfig.model_validate(
                data['Graph']['Ligand']
            ),
            protein=GeneralGraphMolConfig.model_validate(
                data['Graph']['Protein']
            ),
            interaction=GeneralGraphInterConfig.model_validate(
                data['Interaction']
            )
        )