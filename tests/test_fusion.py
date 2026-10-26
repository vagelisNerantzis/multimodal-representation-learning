"""Numeric fusion, missing-modality and shape contracts."""
import torch
import pytest
from multimodal_repr.fusion.mean import mean_fusion
from multimodal_repr.fusion.late import mean_logits


def test_mean_two_modalities():
    torch.testing.assert_close(mean_fusion({'a': torch.ones(2, 3), 'b': torch.full((2, 3), 3.)}),
                               torch.full((2, 3), 2.))


def test_several_modalities_and_insertion_order():
    modalities = {'z': torch.ones(2, 3), 'a': torch.full((2, 3), 3.), 'm': torch.full((2, 3), 5.)}
    output = mean_fusion(modalities)
    torch.testing.assert_close(output, torch.full((2, 3), 3.))
    assert torch.equal(output, mean_fusion(dict(reversed(list(modalities.items())))))


def test_optional_missing_modality():
    x = torch.ones(2, 3)
    torch.testing.assert_close(mean_fusion({'a': x, 'b': None}, allow_missing=True), x)
    with pytest.raises(ValueError, match='Missing modality'):
        mean_fusion({'a': x, 'b': None})


@pytest.mark.parametrize('modalities', [{}, {'a': None}])
def test_no_observed_modalities(modalities):
    with pytest.raises(ValueError, match='Empty|No observed'):
        mean_fusion(modalities, allow_missing=True)


@pytest.mark.parametrize('shape,message', [((2, 4), 'dimensions'), ((3, 3), 'Batch sizes')])
def test_mean_shape_mismatch(shape, message):
    with pytest.raises(ValueError, match=message):
        mean_fusion({'a': torch.ones(2, 3), 'b': torch.ones(shape)})


def test_late_fusion_averages_logits_not_probabilities():
    a, b = torch.tensor([[4., 0.]]), torch.tensor([[0., 2.]])
    output = mean_logits({'a': a, 'b': b})
    torch.testing.assert_close(output, torch.tensor([[2., 1.]]))
    assert not torch.allclose(output.softmax(1), (a.softmax(1) + b.softmax(1)) / 2)


def test_late_fusion_shape_mismatch():
    with pytest.raises(ValueError, match='dimensions'):
        mean_logits({'a': torch.ones(2, 3), 'b': torch.ones(2, 4)})


def test_late_fusion_requires_class_columns():
    with pytest.raises(ValueError, match='two class'):
        mean_logits({'a': torch.ones(2, 1)})


def test_fusion_keeps_gradients():
    a = torch.ones(2, 3, requires_grad=True)
    mean_fusion({'a': a, 'b': a * 2}).sum().backward()
    torch.testing.assert_close(a.grad, torch.full_like(a, 1.5))


def test_mixed_dtypes_rejected():
    with pytest.raises(ValueError, match='dtype and device'):
        mean_fusion({'a': torch.ones(2, 3), 'b': torch.ones(2, 3, dtype=torch.float64)})
