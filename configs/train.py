from dataclasses import dataclass
from utils import options



@dataclass
class TrainConfig:
  
  optimizer:str = options.make_option_field('adam', options=['adam', 'adam_w'])
  seed:int = 42
  n_epochs:int = 100
  learning_rate:float = 0.001
  weight_decay:float = 0.0
  early_stop:bool = options.make_option_field(False, [True, False])
  when_early_stop:int = 100
  loss_func:str = options.make_option_field('mse', ['mse'])
  verbose:int = options.make_option_field(1,[0, 1])
  device: str = options.make_option_field('cpu', ['cpu', 'cuda', 'mps'])
  save_train_log:bool = options.make_option_field(True, [True, False])

  def __post_init__(self):
    options.option_checker(self)


@dataclass
class LoaderConfig:
  
  batch_size:int = 32
  train_frac:float = 0.9
  val_frac:float = 0.05