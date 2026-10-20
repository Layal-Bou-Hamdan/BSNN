"""Inference with functional dropout enabled independently of module train mode."""
from contextlib import contextmanager
import torch


@contextmanager
def inference_mode(model, dropout=False):
    states = [(module, module.training) for module in model.modules()]
    previous = getattr(model, 'mc_dropout', False)
    model.eval()
    model.mc_dropout = dropout
    try:
        with torch.no_grad():
            yield
    finally:
        model.mc_dropout = previous
        for module, training in states:
            module.training = training


def log_probabilities(model, x):
    output = model(x)
    return output[0] if isinstance(output, tuple) else output
