



class ModelError(Exception):
    pass


class WrongGraphConfig(ModelError):

    def __init__(self, msg):
        super().__init__(msg)