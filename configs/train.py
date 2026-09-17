from dataclasses import dataclass
from utils import options
from pydantic import BaseModel, Field

from typing import Literal, Annotated, Self



class TrainConfig(BaseModel):
  
  optimizer: Literal['adam', 'adam_w']
  seed: Annotated[int, Field(default=42, ge=0)]
  n_epochs: Annotated[int, Field(default=100, ge=0)]
  learning_rate: Annotated[float, Field(default=0.00001, ge=0.0)]
  weight_decay: Annotated[float, Field(default=0.00001, ge=0.0)]
  early_stop: Literal[True, False]
  when_early_stop: Annotated[int, Field(default=100, ge=0)]
  loss_func: Literal['mse']
  verbose: Literal[0, 1]
  device: Literal['cpu', 'cuda', 'mps']
  show_test_metrics: Literal[True, False]
  save_train_log: Literal[True, False]


class LoaderConfig(BaseModel):
  
  batch_size: Annotated[int, Field(default=32, ge=0)]
  train_frac: Annotated[float, Field(default=0.9, ge=0.0, le=1)]
  val_frac: Annotated[float, Field(default=0.9, ge=0.0, le=1)]



class TrainLoaderConfig(BaseModel):
  train: TrainConfig | None=None
  loader: LoaderConfig | None=None

  @classmethod
  def load_data(
    cls, 
    data: dict
  ):

    return cls(
      train=data['TrainingSettings'],
      loader=data['LoaderSettings']
    )