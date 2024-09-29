import pickle

import numpy as np
import torch
import torch.nn as nn


def c2_msra_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def convert_numpy_to_torch(state_dict):
    for key, value in state_dict.items():
        state_dict[key] = torch.from_numpy(value)

    return state_dict


def load_model(model: nn.Module, weights_path: str):
    with open(weights_path, "rb") as f:
        unpickled_weights = pickle.load(f)
    state_dict = unpickled_weights["model"]
    tensor_state_dict = convert_numpy_to_torch(state_dict)
    model.load_state_dict(tensor_state_dict, strict=True)
    return model


def compare_weights(unweighted, weighted) -> bool:
    """Asserts that all weights have been loaded correctly"""

    unchanged = 0
    for (name_u, param_u), (name_w, param_w) in zip(unweighted, weighted):
        if not torch.equal(param_u, param_w):
            # print(f"Parameter '{name_u}' has changed.")
            pass
        else:
            print(f"Parameter '{name_u}' is unchanged.")
            unchanged += 1

    return True if unchanged == 0 else False


def get_cloned_parameters(model):
    return [(name, param.clone()) for name, param in model.named_parameters()]
