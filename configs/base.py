from pydantic import ValidationError
from exceptions.config import check_validation_error_exc

from functools import wraps



def config_validation_wrapp(func):
    @wraps(func)
    def validate(*args, **kwargs):
        try:
            config = func(*args, **kwargs)
            return config
        except ValidationError as exc:
            print(exc)
            check_validation_error_exc(exc)

    return validate


        