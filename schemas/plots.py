from dataclasses import dataclass
import numpy as np



@dataclass
class ModelPredictions:

    y_true: np.ndarray = None
    y_pred: np.ndarray = None