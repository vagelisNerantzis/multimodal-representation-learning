"""Retrieval correctness is ID-based, with explicit gallery semantics."""
import torch
import pytest
from multimodal_repr.evaluation.retrieval import cosine_similarity_matrix, top1_cross_modal_accuracy


def test_perfect_retrieval_with_reordered_gallery():
    x = torch.eye(3)
    order = torch.tensor([2, 0, 1])
    assert top1_cross_modal_accuracy(x, x[order], ['a', 'b', 'c'], ['c', 'a', 'b']) == 1


def test_incorrect_retrieval_with_wrong_identity_assignment():
    x = torch.eye(3)
    assert top1_cross_modal_accuracy(x, x, ['a', 'b', 'c'], ['b', 'c', 'a']) == 0


def test_cosine_shape_and_numeric_values():
    q = torch.tensor([[1., 0.], [0., 2.]])
    g = torch.tensor([[3., 0.], [0., 1.], [1., 1.]])
    result = cosine_similarity_matrix(q, g)
    assert result.shape == (2, 3)
    torch.testing.assert_close(result[:, :2], torch.eye(2))


def test_duplicate_gallery_ids_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        top1_cross_modal_accuracy(torch.eye(2), torch.eye(2), ['a', 'b'], ['a', 'a'])


def test_query_missing_from_gallery():
    with pytest.raises(ValueError, match='occur in the gallery'):
        top1_cross_modal_accuracy(torch.eye(2), torch.eye(2), ['a', 'b'], ['a', 'c'])


def test_retrieval_dimension_mismatch():
    with pytest.raises(ValueError, match='dimensions'):
        cosine_similarity_matrix(torch.ones(2, 3), torch.ones(2, 4))


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_embeddings_rejected(bad):
    x = torch.eye(2)
    x[0, 0] = bad
    with pytest.raises(ValueError, match='NaN or infinite'):
        cosine_similarity_matrix(x, torch.eye(2))


def test_zero_and_empty_embeddings_rejected():
    with pytest.raises(ValueError, match='zero embedding'):
        cosine_similarity_matrix(torch.zeros(2, 3), torch.ones(2, 3))
    with pytest.raises(ValueError, match='nonempty shape'):
        cosine_similarity_matrix(torch.empty(0, 3), torch.ones(2, 3))
