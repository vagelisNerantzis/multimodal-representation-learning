"""Shared validation with explicit physical sample identity and tensor semantics."""
from collections.abc import Mapping, Sequence
import torch
from torch import Tensor


def validate_embeddings(value: Tensor, *, name: str = "embeddings", nonzero: bool = False) -> Tensor:
    """Require a nonempty finite floating [N,D] tensor; optionally reject zero rows."""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError(f"{name} must have nonempty shape [N, D]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains NaN or infinite values")
    if nonzero and (value.abs().amax(dim=1) == 0).any().item():
        raise ValueError(f"{name} contains a zero embedding; cosine direction is undefined")
    return value


def validate_pair(a: Tensor, b: Tensor, *, matching_batch: bool = True, nonzero: bool = False) -> None:
    """Require common dimensions, dtype and device, and optionally matching rows."""
    validate_embeddings(a, name="modality A", nonzero=nonzero)
    validate_embeddings(b, name="modality B", nonzero=nonzero)
    if a.shape[1] != b.shape[1]:
        raise ValueError("Embedding dimensions differ")
    if matching_batch and a.shape[0] != b.shape[0]:
        raise ValueError("Batch sizes differ")
    if a.dtype != b.dtype or a.device != b.device:
        raise ValueError("Tensors must share dtype and device")


def validate_sample_ids(ids: Sequence[str], count: int, *, unique: bool = False) -> tuple[str, ...]:
    """Require explicit, nonempty string identities, preserving their supplied order."""
    if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
        raise TypeError("sample_ids must be a sequence of strings")
    if len(ids) != count:
        raise ValueError("sample_ids length must match embedding rows")
    for sample_id in ids:
        if not isinstance(sample_id, str):
            raise TypeError("Every sample ID must be a string")
        if not sample_id or sample_id != sample_id.strip():
            raise ValueError("Sample IDs must be nonempty and have no surrounding whitespace")
    result = tuple(ids)
    if unique and len(set(result)) != len(result):
        raise ValueError("Duplicate sample IDs are ambiguous; aggregate observations first")
    return result


def modality_tensors(modalities: Mapping[str, Tensor | None], *, allow_missing: bool = False) -> list[Tensor]:
    """Validate complete-modality tensors in sorted-name order; never infer row identity."""
    if not isinstance(modalities, Mapping):
        raise TypeError("modalities must be a mapping from names to tensors")
    if not modalities:
        raise ValueError("Empty modality set")
    if any(not isinstance(k, str) or not k.strip() for k in modalities):
        raise ValueError("Modality names must be nonempty strings")
    tensors = []
    for name in sorted(modalities):
        value = modalities[name]
        if value is None:
            if not allow_missing:
                raise ValueError(f"Missing modality {name}; set allow_missing=True to omit it")
            continue
        validate_embeddings(value, name=name)
        if tensors:
            validate_pair(tensors[0], value)
        tensors.append(value)
    if not tensors:
        raise ValueError("No observed modalities remain")
    return tensors


def unit_vectors(value: Tensor) -> Tensor:
    """Stable L2 normalization, rejecting undefined zero vectors before normalization."""
    validate_embeddings(value, nonzero=True)
    value = value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
    # Scaling before the norm prevents overflow for large finite coordinates.
    scaled = value / value.abs().amax(dim=1, keepdim=True)
    return scaled / torch.linalg.vector_norm(scaled, dim=1, keepdim=True)
