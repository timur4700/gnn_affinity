from dataclasses import dataclass
from typing import Any


@dataclass
class DataSetSplited:
    train: list[Any] = None
    val: list[Any] = None
    test: list[Any] = None