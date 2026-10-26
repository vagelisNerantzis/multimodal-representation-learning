"""Symmetric InfoNCE for one paired representation per physical sample."""
from collections.abc import Sequence
import math
from numbers import Real
import torch
from torch import Tensor
from torch.nn import functional as F
from ..validation.inputs import validate_pair, validate_sample_ids, unit_vectors


def symmetric_info_nce(a: Tensor, b: Tensor, *, temperature: float = 0.1,
                       sample_ids_a: Sequence[str] | None = None,
                       sample_ids_b: Sequence[str] | None = None) -> Tensor:
    """Mean A-to-B and B-to-A cross entropy over normalized paired [B,D] vectors.

    Row i must identify the same physical sample in both modalities. Optional IDs
    verify this contract and reject duplicate samples (false negatives). Without
    IDs this invariant is the caller's responsibility. Other samples are negatives,
    even if they have the same class; no class-aware masking is implemented.
    """
    if isinstance(temperature, bool) or not isinstance(temperature, Real):
        raise TypeError("temperature must be a finite positive real number")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    validate_pair(a, b, nonzero=True)
    if a.shape[0] < 2:
        raise ValueError("Contrastive alignment requires at least two paired samples")
    if (sample_ids_a is None) != (sample_ids_b is None):
        raise ValueError("Supply sample IDs for both modalities or neither")
    if sample_ids_a is not None:
        ids_a = validate_sample_ids(sample_ids_a, len(a), unique=True)
        ids_b = validate_sample_ids(sample_ids_b, len(b), unique=True)
        if ids_a != ids_b:
            raise ValueError("Paired sample IDs must match row by row")
    logits = unit_vectors(a) @ unit_vectors(b).T / float(temperature)
    if not torch.isfinite(logits).all().item():
        raise ValueError("Temperature is too small for a finite similarity matrix")
    targets = torch.arange(len(a), device=a.device)
    loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
    if not torch.isfinite(loss).item():
        raise ValueError("Contrastive loss is nonfinite at this temperature")
    return loss
