"""Differentiable observation-to-physical-sample mean aggregation."""
from collections.abc import Sequence
from dataclasses import dataclass
import torch
from torch import Tensor
from ..validation.inputs import validate_embeddings, validate_sample_ids


@dataclass(frozen=True)
class SampleEmbeddings:
    """Explicit row-to-sample mapping, ordered by first appearance in the input."""
    sample_ids: tuple[str, ...]
    embeddings: Tensor


def aggregate_samples(embeddings: Tensor, sample_ids: Sequence[str]) -> SampleEmbeddings:
    """Average all observation rows for each ID, preserving first-appearance order.

    Call separately per modality and partition. All observations get equal weight;
    if crops per image vary, aggregate within images before averaging images.
    No labels, partitions or latent identities are inferred from embedding values.
    """
    validate_embeddings(embeddings)
    ids = validate_sample_ids(sample_ids, len(embeddings))
    groups: dict[str, list[int]] = {}
    for i, sample_id in enumerate(ids):
        groups.setdefault(sample_id, []).append(i)
    values = torch.stack([embeddings[indices].mean(dim=0) for indices in groups.values()])
    validate_embeddings(values, name="sample embeddings")
    return SampleEmbeddings(tuple(groups), values)
