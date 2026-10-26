"""ID-aware cosine retrieval between two modality-specific embedding tables."""
from collections.abc import Sequence
import torch
from torch import Tensor
from ..validation.inputs import validate_pair, validate_sample_ids, unit_vectors


def cosine_similarity_matrix(queries: Tensor, gallery: Tensor) -> Tensor:
    """Return [N,M] pairwise cosine similarities for nonzero shared-D vectors."""
    validate_pair(queries, gallery, matching_batch=False, nonzero=True)
    return (unit_vectors(queries) @ unit_vectors(gallery).T).clamp(-1, 1)


def top1_cross_modal_accuracy(queries: Tensor, gallery: Tensor,
                             query_ids: Sequence[str], gallery_ids: Sequence[str]) -> float:
    """Measure whether each query's nearest gallery row has the same physical ID.

    IDs must be unique in each table; aggregate repeated observations first.
    Gallery order may differ and extra gallery distractors are allowed, but every
    query must have a corresponding gallery identity. Ties use first gallery row.
    This is A-to-B retrieval; call with swapped inputs to evaluate B-to-A.
    """
    similarities = cosine_similarity_matrix(queries, gallery)
    qids = validate_sample_ids(query_ids, len(queries), unique=True)
    gids = validate_sample_ids(gallery_ids, len(gallery), unique=True)
    if not set(qids).issubset(gids):
        raise ValueError("Every query sample ID must occur in the gallery")
    nearest = similarities.argmax(dim=1).tolist()
    return sum(qid == gids[index] for qid, index in zip(qids, nearest)) / len(qids)
