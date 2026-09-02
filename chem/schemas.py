from dataclasses import dataclass
import numpy as np




@dataclass
class MolGraph:
    x: np.ndarray = None
    pos: np.ndarray = None
    edge_index: np.ndarray = None
    edge_type: np.ndarray = None
    y: float = None