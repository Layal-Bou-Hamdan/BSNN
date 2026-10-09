"""Differentiable Householder products using only PyTorch tensor operations."""
import torch


def householder_product(a):
    """Return H_0 ... H_(d-1), with unit-leading reflectors from a's columns.

    Only the strict lower triangle is parameterized. Each reflector vector has
    zeros above its diagonal and a leading one, so its norm cannot vanish.
    This is a product of reflectors, not a QR decomposition of a.
    """
    if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError('Expected square matrices')
    d = a.shape[-1]
    eye = torch.eye(d, dtype=a.dtype, device=a.device)
    q = eye.expand(a.shape)
    vectors = a.tril(-1) + eye
    for i in range(d):
        v = vectors[..., :, i:i+1]
        qv = q @ v
        q = q - (2 / v.square().sum(dim=-2, keepdim=True)) * qv @ v.transpose(-2, -1)
    return q
