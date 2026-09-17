class DatasetError(Exception):
    pass


class DatasetMetadataNotFound(Exception):

    message = "Dataset's metadata was not found. Prepare dataset with command <gnn-affinity prepare>"

    def __init__(self):
        super().__init__(self.message)