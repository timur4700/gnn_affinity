from pydantic import BaseModel, Field
from typing import Annotated, Literal



class ModelSettings(BaseModel):

    # General settings
    n_rbf: Annotated[int, Field(default=32, ge=1)]
    hidden_dim: Annotated[int, Field(default=64, ge=1)]
    output_dim: Annotated[int, Field(default=1, ge=1)]
    cutoff: Annotated[float, Field(default=5.0, gt=0)]

    # Ligand encoder
    ligand_mpnn: Literal['gin', 'gcn', 'gat', 'egnn']= "gin"
    ligand_mpnn_n_layers: Annotated[int, Field(default=2, ge=1)]

    # Protein encoder
    protein_mpnn: Literal['gin', 'gcn', 'gat', 'egnn'] = "gin"
    protein_mpnn_n_layers: Annotated[int, Field(default=2, ge=1)]

    # Protein-ligand interaction encoder
    interaction_mpnn: Literal['gin', 'gcn', 'gat', 'egnn'] = "egnn"
    interaction_mpnn_n_layers: Annotated[int, Field(default=4, ge=1)]

    igand_cutoff: Annotated[float, Field(default=3.0, ge=0)]
    protein_cutoff: Annotated[float, Field(default=3.0, ge=0)]
    interaction_cutoff: Annotated[float, Field(default=5.0, ge=0)]



class ModelSettingsOld(BaseModel):
    n_rbf: Annotated[int, Field(default=32, ge=1)]
    hidden_dim:  Annotated[int, Field(default=64, ge=1)]
    output_dim:  Annotated[int, Field(default=1, ge=1)]
    cutoff:  Annotated[float, Field(default=5.0, ge=0.0)]
    dropout: Annotated[float, Field(default=0.1, ge=0.0, le=1.0)]
    n_gine:  Annotated[int, Field(default=2, ge=1)]
    n_egnn: Annotated[int, Field(default=2, ge=1)]
    normalization: Literal['batchnorm', 'graphnorm', 'layernorm'] = 'batchnorm'