###
# Author: Kai Li
# Date: 2022-02-12 15:16:35
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-09-15 11:32:37
###
from .bsrnn import BSRNN
from .afrcnn import AFRCNN
from .tdanet import TDANet
from .gc3_network import TasNet
from .sepformer import Sepformer
from .convtasnet import ConvTasNet
from .dprnn_old import DPRNNTasNet
from .sandglasset import Sandglasset


__all__ = [
    "BSRNN",
    "TDANet",
    "AFRCNN",
    "TasNet",
    "Sepformer",
    "ConvTasNet",
    "DPRNNTasNet",
    "Sandglasset",
]


def register_model(custom_model):
    """Register a custom model, gettable with `models.get`.

    Args:
        custom_model: Custom model to register.

    """
    if custom_model.__name__ in globals().keys() or custom_model.__name__.lower() in globals().keys():
        raise ValueError(f"Model {custom_model.__name__} already exists. Choose another name.")
    globals().update({custom_model.__name__: custom_model})


def get(identifier):
    """Returns an model class from a string (case-insensitive).

    Args:
        identifier (str): the model name.

    Returns:
        :class:`torch.nn.Module`
    """
    if isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Could not interpret model name : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret model name : {str(identifier)}")
