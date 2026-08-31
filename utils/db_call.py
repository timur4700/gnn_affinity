from models.build.model_register import MODEL_REGISTER
from datasets.register import DATASET_REGISTER

from typing import Any
import importlib

class DBCaller():
    def __init__(self):

        self.register: dict[str, Any] = None

    def __call__(self, name) -> Any | None:
        return self.register.get(name, None)


class DatasetDB(DBCaller):
    def __init__(self):
        super().__init__()

        self.register = DATASET_REGISTER

class ModelDB(DBCaller):
    def __init__(self):

        super().__init__()
        self.register = MODEL_REGISTER