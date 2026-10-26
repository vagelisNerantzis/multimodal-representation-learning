"""Equal-weight embedding fusion; shared coordinates and row identity are prerequisites."""
from collections.abc import Mapping
import torch
from torch import Tensor
from ..validation.inputs import modality_tensors, validate_embeddings


def mean_fusion(modalities: Mapping[str, Tensor | None], *, allow_missing: bool = False) -> Tensor:
    """Average aligned [B,D] embeddings in deterministic sorted-modality order.

    None can omit an entire modality when explicitly allowed. Per-row missingness
    is not supported. No projection, normalization or identity join is implicit.
    """
    tensors = modality_tensors(modalities, allow_missing=allow_missing)
    # Divide before summing to reduce overflow of finite large inputs.
    result = torch.stack([x / len(tensors) for x in tensors]).sum(dim=0)
    return validate_embeddings(result, name="fused embeddings")
