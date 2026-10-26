"""Prediction-level mean-logit fusion, not probability averaging."""
from collections.abc import Mapping
from torch import Tensor
from .mean import mean_fusion


def mean_logits(modalities: Mapping[str, Tensor | None], *, allow_missing: bool = False) -> Tensor:
    """Average [B,C] logits with a common ordered class schema and sample order.

    Apply softmax after this function if probabilities are needed. Logits must
    have comparable scales; validation cannot infer class meanings or calibration.
    """
    result = mean_fusion(modalities, allow_missing=allow_missing)
    if result.shape[1] < 2:
        raise ValueError("Mean-logit fusion requires at least two class columns")
    return result
