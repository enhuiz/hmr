from .utils import get_class
from .cyclegan import CycleGAN
from .munit import MUNIT
from .seq2seq import Seq2Seq


def get_model(opts):
    model_list = [CycleGAN, MUNIT, Seq2Seq]
    return get_class(model_list, opts.model)(opts)
