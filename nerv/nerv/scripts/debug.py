"""Debug file loading params and models"""

import os
import sys
import importlib

import torch

from models import build_model
from method import build_method
from datasets import build_dataset


def load_params(params):
    if params.endswith('.py'):
        params = params[:-3]
    sys.path.append(os.path.dirname(params))
    params = importlib.import_module(os.path.basename(params))
    params = params.SlotAttentionParams()
    model = build_model(params)
    return params, model
