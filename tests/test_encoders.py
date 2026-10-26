"""Synthetic vector encoders and package contracts."""
import importlib.metadata
import torch
import pytest
import multimodal_repr
from multimodal_repr.encoders.base import BaseEncoder, IdentityEncoder, LinearEncoder


def test_package_version():
    assert multimodal_repr.__version__ == '0.1.0'
    assert importlib.metadata.version('multimodal-representation-learning') == '0.1.0'


def test_abstract_interface():
    with pytest.raises(TypeError):
        BaseEncoder(3)


def test_identity_preserves_values_and_order():
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    encoder = IdentityEncoder(3)
    assert isinstance(encoder, torch.nn.Module)
    assert encoder.embedding_dim == 3
    assert encoder(x) is x


def test_linear_encoder_known_deterministic_projection():
    encoder = LinearEncoder(3, 2, bias=False)
    with torch.no_grad():
        encoder.projection.weight.copy_(torch.tensor([[1., 0., 0.], [0., 2., 0.]]))
    x = torch.tensor([[2., 3., 4.], [5., 6., 7.]])
    expected = torch.tensor([[2., 6.], [5., 12.]])
    torch.testing.assert_close(encoder(x), expected)
    torch.testing.assert_close(encoder(x), encoder(x))


def test_linear_encoder_backward():
    encoder = LinearEncoder(3, 2)
    encoder(torch.ones(4, 3)).sum().backward()
    assert encoder.projection.weight.grad is not None
    assert torch.isfinite(encoder.projection.weight.grad).all()


@pytest.mark.parametrize('dimension', [0, -1])
def test_invalid_embedding_dimension(dimension):
    with pytest.raises(ValueError, match='positive integer'):
        IdentityEncoder(dimension)


def test_encoder_input_dimension_mismatch():
    with pytest.raises(ValueError, match='Input dimension'):
        IdentityEncoder(3)(torch.ones(2, 4))
