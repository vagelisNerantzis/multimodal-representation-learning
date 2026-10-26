"""Encoders accept modality-specific inputs and emit one embedding per input row."""
from abc import ABC, abstractmethod
from typing import Any
import torch
from torch import Tensor, nn
from ..validation.inputs import validate_embeddings


class BaseEncoder(nn.Module, ABC):
    """Subclass forward for any modality; return finite floating [B, embedding_dim].

    Input structure is modality-specific and need not be an image or tensor.
    Row/sample identity is carried by the caller, never inferred by this module.
    """
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if type(embedding_dim) is not int or embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer")
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(self, inputs: Any) -> Tensor:
        """Encode a batch without reordering its physical identities."""
        raise NotImplementedError


class IdentityEncoder(BaseEncoder):
    """Validate already computed [B,D] embeddings without altering them."""
    def forward(self, inputs: Tensor) -> Tensor:
        validate_embeddings(inputs, name="encoder input")
        if inputs.shape[1] != self.embedding_dim:
            raise ValueError("Input dimension differs from embedding_dim")
        return inputs


class LinearEncoder(BaseEncoder):
    """Small trainable vector encoder; initialization follows PyTorch's RNG."""
    def __init__(self, input_dim: int, embedding_dim: int, *, bias: bool = True) -> None:
        super().__init__(embedding_dim)
        if type(input_dim) is not int or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        self.input_dim = input_dim
        self.projection = nn.Linear(input_dim, embedding_dim, bias=bias)

    def forward(self, inputs: Tensor) -> Tensor:
        validate_embeddings(inputs, name="encoder input")
        if inputs.shape[1] != self.input_dim:
            raise ValueError("Input dimension differs from input_dim")
        if inputs.dtype != self.projection.weight.dtype or inputs.device != self.projection.weight.device:
            raise ValueError("Encoder input and parameters must share dtype and device")
        result = self.projection(inputs)
        return validate_embeddings(result, name="encoder output")
