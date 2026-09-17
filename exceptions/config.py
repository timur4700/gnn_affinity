from pydantic import ValidationError

class WrongChoiceError(Exception):
    pass


def check_validation_error_exc(
        exc: ValidationError
):

    for error in exc.errors():
        if error['type'] == 'literal_error':
            raise WrongChoiceError(
                f"Unrecognized option: {error['input']}\n"\
                f"Available Options for field {error['loc'][0]}: {error['ctx']['expected']}"
            )

        else:
            raise exc