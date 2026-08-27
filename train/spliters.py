from train.utils import general_spliter
from datasets.pdbbind.spliter import pdbbind_spliter



SPLITERS = {
    'default': general_spliter,
    'pdbbind_dataset': pdbbind_spliter
}