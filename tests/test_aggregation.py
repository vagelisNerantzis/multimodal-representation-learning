"""Observation grouping preserves physical identity and gradients."""
import torch
import pytest
from multimodal_repr.aggregation.sample import aggregate_samples


def test_group_means_and_first_appearance_order():
    x = torch.tensor([[1., 3.], [10., 20.], [3., 5.], [20., 40.]])
    result = aggregate_samples(x, ['b', 'a', 'b', 'a'])
    assert result.sample_ids == ('b', 'a')
    torch.testing.assert_close(result.embeddings, torch.tensor([[2., 4.], [15., 30.]]))


def test_aggregation_gradients_respect_group_sizes():
    x = torch.ones(3, 2, requires_grad=True)
    aggregate_samples(x, ['a', 'a', 'b']).embeddings.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[.5, .5], [.5, .5], [1., 1.]]))


@pytest.mark.parametrize('ids,error', [(['s'], ValueError), (['s', ''], ValueError), (['s', None], TypeError)])
def test_invalid_or_missing_sample_ids(ids, error):
    with pytest.raises(error, match='sample_ids length|Sample IDs|sample ID'):
        aggregate_samples(torch.ones(2, 3), ids)
