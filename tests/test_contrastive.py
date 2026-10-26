"""Controlled paired identity, stable symmetric loss and real autograd."""
import torch
import pytest
from multimodal_repr.losses.contrastive import symmetric_info_nce


def test_finite_positive_scalar_loss():
    x = torch.eye(4)
    loss = symmetric_info_nce(x, x)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss > 0


def test_backward_has_finite_nonzero_gradients_in_both_modalities():
    a = torch.tensor([[1., .2, .1], [.1, 1., .3], [.2, .1, 1.]], requires_grad=True)
    b = torch.tensor([[.9, .3, .1], [.2, 1., .1], [.1, .2, 1.]], requires_grad=True)
    symmetric_info_nce(a, b, temperature=.2).backward()
    for x in (a, b):
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0


def test_aligned_pairing_outperforms_shuffled_pairing():
    x = torch.eye(4)
    assert symmetric_info_nce(x, x) < symmetric_info_nce(x, x.roll(1, dims=0))


def test_symmetry_and_internal_normalization():
    a = torch.tensor([[1., .2], [.1, 1.]])
    b = torch.tensor([[.9, .3], [.2, 1.]])
    torch.testing.assert_close(symmetric_info_nce(a, b), symmetric_info_nce(b, a))
    torch.testing.assert_close(symmetric_info_nce(a, b), symmetric_info_nce(a * 10, b * 3))


@pytest.mark.parametrize('temperature', [0., -1., float('nan'), float('inf')])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError, match='temperature'):
        symmetric_info_nce(torch.eye(3), torch.eye(3), temperature=temperature)


@pytest.mark.parametrize('shape', [(2, 4), (3, 3)])
def test_contrastive_shape_mismatch(shape):
    with pytest.raises(ValueError, match='dimensions|Batch sizes'):
        symmetric_info_nce(torch.ones(2, 3), torch.ones(shape))


def test_single_sample_is_not_contrastive():
    with pytest.raises(ValueError, match='two paired samples'):
        symmetric_info_nce(torch.ones(1, 3), torch.ones(1, 3))


def test_explicit_paired_ids_and_one_sided_failure():
    x = torch.eye(2)
    assert torch.isfinite(symmetric_info_nce(x, x, sample_ids_a=['s1', 's2'], sample_ids_b=['s1', 's2']))
    with pytest.raises(ValueError, match='both modalities'):
        symmetric_info_nce(x, x, sample_ids_a=['s1', 's2'])


@pytest.mark.parametrize('ids_b', [['s2', 's1'], ['s1', 's1']])
def test_ambiguous_or_mispaired_ids(ids_b):
    with pytest.raises(ValueError, match='row by row|Duplicate'):
        symmetric_info_nce(torch.eye(2), torch.eye(2), sample_ids_a=['s1', 's2'], sample_ids_b=ids_b)


@pytest.mark.parametrize('dtype,scale', [
    (torch.float32, 1e-40),
    (torch.float32, 1e-30),
    (torch.float32, 1e-10),
    (torch.float32, 1.),
    (torch.float32, 1e30),
    (torch.float64, 1e-200),
])
def test_near_zero_and_ordinary_embeddings_have_finite_gradients(dtype, scale):
    a = ((torch.eye(3, dtype=dtype) + .1) * scale).requires_grad_()
    b = ((torch.eye(3, dtype=dtype) + .2) * scale).requires_grad_()
    loss = symmetric_info_nce(a, b, temperature=.2)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    for value in (a, b):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().max() > 0


def test_subnormal_float32_matches_explicit_norm_floor_reference():
    from torch.nn import functional as F
    a = ((torch.eye(3) + .1) * 1e-40).requires_grad_()
    b = ((torch.eye(3) + .2) * 1e-40).requires_grad_()
    eps = torch.finfo(a.dtype).tiny ** .5
    logits = F.normalize(a, dim=1, eps=eps) @ F.normalize(b, dim=1, eps=eps).T / .2
    targets = torch.arange(3)
    expected = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
    actual = symmetric_info_nce(a, b, temperature=.2)
    torch.testing.assert_close(actual, expected)
    expected_grads = torch.autograd.grad(expected, (a, b))
    actual_grads = torch.autograd.grad(actual, (a, b))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_ordinary_loss_and_gradients_match_unregularized_cosine(dtype):
    a = torch.tensor([[1., .2, .1], [.1, 1., .3], [.2, .1, 1.]], dtype=dtype, requires_grad=True)
    b = torch.tensor([[.9, .3, .1], [.2, 1., .1], [.1, .2, 1.]], dtype=dtype, requires_grad=True)
    from torch.nn import functional as F
    u = a / torch.linalg.vector_norm(a, dim=1, keepdim=True)
    v = b / torch.linalg.vector_norm(b, dim=1, keepdim=True)
    logits = u @ v.T / .2
    targets = torch.arange(3)
    expected = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
    actual = symmetric_info_nce(a, b, temperature=.2)
    torch.testing.assert_close(actual, expected)
    expected_grads = torch.autograd.grad(expected, (a, b))
    actual_grads = torch.autograd.grad(actual, (a, b))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_contrastive_gradient_matches_finite_differences():
    a = torch.tensor([[1., .2], [.1, 1.]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[.9, .3], [.2, 1.]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, y: symmetric_info_nce(x, y, temperature=.2), (a, b))


@pytest.mark.parametrize('zero_modality', ['a', 'b'])
def test_exact_zero_embeddings_remain_invalid(zero_modality):
    a, b = torch.eye(3), torch.eye(3)
    (a if zero_modality == 'a' else b)[0].zero_()
    with pytest.raises(ValueError, match='zero embedding'):
        symmetric_info_nce(a, b)
