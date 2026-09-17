from dataclasses import dataclass, field
from typing import Literal, Annotated, Self
from utils import options
from pydantic import BaseModel, Field
from configs.base import config_validation_wrapp


class Formats(BaseModel):
    ligand: Literal['sdf', 'mol2']


class LigandConfig(BaseModel):
    sanitize: Literal[True, False]


class ProteinConfig(BaseModel):
    extract_pocket: Literal[True, False]
    extract_method: Literal['atom', 'cog', 'com']

    pocket_cutoff: Annotated[float | int, Field(ge=0.0)] = 10.0
    sanitize: Literal[True, False]


class MolConfig(BaseModel):
    formats: Formats
    ligand: LigandConfig
    protein: ProteinConfig

    @classmethod
    @config_validation_wrapp
    def load_data(cls, data) -> Self:

        return MolConfig(
            formats=Formats.model_validate(
                data['Formats']
                ),
            ligand=LigandConfig.model_validate(
                data['Ligand']
            ),
            protein=ProteinConfig.model_validate(
                data['Protein']
            )            
        )


class PDBbindMolConfig(MolConfig):
    prot_source: str

    @classmethod
    @config_validation_wrapp
    def load_data(cls, data) -> Self:
        mol_base = super().load_data(data)

        return cls(
            formats=mol_base.formats,
            ligand=mol_base.ligand,
            protein=mol_base.protein,
            prot_source=data["ProteinSource"],
        )