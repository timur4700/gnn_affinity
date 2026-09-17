from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Annotated, Literal


class ModelSettings(BaseModel):
    n_rbf: Annotated[int, Field(default=32, ge=1)]
    hidden_dim:  Annotated[int, Field(default=64, ge=1)]
    output_dim:  Annotated[int, Field(default=1, ge=1)]
    cutoff:  Annotated[float, Field(default=5.0, ge=0.0)]
    dropout: Annotated[float, Field(default=0.1, ge=0.0, le=1.0)]
    n_gine:  Annotated[int, Field(default=2, ge=1)]
    n_egnn: Annotated[int, Field(default=2, ge=1)]
    normalization: Literal['batchnorm', 'graphnorm', 'layernorm'] = 'batchnorm'