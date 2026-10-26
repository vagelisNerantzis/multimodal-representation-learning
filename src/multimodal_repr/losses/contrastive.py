"""Symmetric InfoNCE for one paired representation per physical sample."""
from collections.abc import Sequence
import math
from numbers import Real
import torch
from torch import Tensor
from torch.nn import functional as F
from ..validation.inputs import validate_pair, validate_sample_ids


def _normalize_for_alignment(value: Tensor) -> Tensor:
    """Compute x/max(||x||_2, eps) without dividing by a subnormal row scale."""
    value = value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
    # sqrt(tiny) is the dtype's squared-norm underflow threshold. Unlike an
    # arbitrary epsilon, it follows the arithmetic precision used by the loss.
    eps = math.sqrt(torch.finfo(value.dtype).tiny)
    # Rescale only large rows. Divisors stay >=1; their backward cannot create
    # the unrepresentable reciprocal that caused NaNs for subnormal inputs.
    scale = value.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    return F.normalize(value / scale, p=2, dim=1, eps=eps)


def symmetric_info_nce(a: Tensor, b: Tensor, *, temperature: float = 0.1,
                       sample_ids_a: Sequence[str] | None = None,
                       sample_ids_b: Sequence[str] | None = None) -> Tensor:
    """Mean A-to-B and B-to-A cross entropy over normalized paired [B,D] vectors.

    Row i must identify the same physical sample in both modalities. Optional IDs
    verify this contract and reject duplicate samples (false negatives). Without
    IDs this invariant is the caller's responsibility. Other samples are negatives,
    even if they have the same class; no class-aware masking is implemented.

    L2 norms use an explicit floor sqrt(finfo(compute_dtype).tiny), about
    1.084e-19 in float32. Below this floor normalization is epsilon-regularized,
    ensuring finite near-zero gradients; ordinary norms keep the cosine loss.
    Half/bfloat16 computation is promoted to float32. Exactly zero rows remain
    invalid. No embedding coordinates or gradients are clipped.
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
    logits = _normalize_for_alignment(a) @ _normalize_for_alignment(b).T / float(temperature)
    if not torch.isfinite(logits).all().item():
        raise ValueError("Temperature is too small for a finite similarity matrix")
    targets = torch.arange(len(a), device=a.device)
    loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
    if not torch.isfinite(loss).item():
        raise ValueError("Contrastive loss is nonfinite at this temperature")
    return loss
