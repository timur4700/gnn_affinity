
from pathlib import Path

class MetaDataError(Exception):
    pass


class MetaDataNotFound(MemoryError):
    message = '{} metadata was not found in {}'

    def __init__(
            self, 
            metadata_name, 
            directory_path: Path=None
            ):
        
        super().__init__(self.message.format(metadata_name, str(directory_path)))