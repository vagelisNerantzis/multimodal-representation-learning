"""Known synthetic alignment illustrates mechanics, not learned generalization.

Run after installing the package: python examples/synthetic_multimodal_demo.py
No files, datasets, checkpoints, downloads, or GPU are used.
"""
import argparse
import json
import numpy as np
import torch
from multimodal_repr.encoders.base import LinearEncoder
from multimodal_repr.aggregation.sample import aggregate_samples
from multimodal_repr.fusion.mean import mean_fusion
from multimodal_repr.losses.contrastive import symmetric_info_nce
from multimodal_repr.evaluation.retrieval import top1_cross_modal_accuracy


def main() -> None:
    """Build paired repeated views, align, fuse, aggregate and retrieve on CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embedding-dim', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.1)
    args = parser.parse_args()
    if args.embedding_dim < 6:
        parser.error('embedding-dim must be at least 6 for the six synthetic samples')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    samples, views, dim = 6, 3, args.embedding_dim
    identities = [f'synthetic_{i:02d}' for i in range(samples) for _ in range(views)]
    latent = torch.eye(dim)[:samples].repeat_interleave(views, dim=0)
    # Known shared signal, different input dimensions/scales, independent view noise.
    observations_a = latent + torch.tensor(rng.normal(0, 0.01, latent.shape), dtype=torch.float32)
    observations_b = torch.cat([2 * latent + torch.randn_like(latent) * 0.02,
                                torch.zeros(samples * views, 2)], dim=1)
    encoder_a = LinearEncoder(dim, dim, bias=False)
    encoder_b = LinearEncoder(dim + 2, dim, bias=False)
    # Handcrafted alignment is intentional: no training or held-out claim is made.
    with torch.no_grad():
        encoder_a.projection.weight.copy_(torch.eye(dim))
        encoder_b.projection.weight.zero_()
        encoder_b.projection.weight[:, :dim].copy_(torch.eye(dim) / 2)
    a, b = encoder_a(observations_a), encoder_b(observations_b)
    sample_a = aggregate_samples(a, identities)
    sample_b = aggregate_samples(b, identities)
    # Aggregate repeated views before InfoNCE to avoid duplicate-sample negatives.
    loss = symmetric_info_nce(sample_a.embeddings, sample_b.embeddings,
                             sample_ids_a=sample_a.sample_ids, sample_ids_b=sample_b.sample_ids,
                             temperature=args.temperature)
    loss.backward()
    fused_views = mean_fusion({'sensor_a': a, 'sensor_b': b})
    fused_samples = aggregate_samples(fused_views, identities)
    retrieval = top1_cross_modal_accuracy(sample_a.embeddings, sample_b.embeddings,
                                          sample_a.sample_ids, sample_b.sample_ids)
    print(json.dumps({'data': 'synthetic only; known alignment, not trained performance',
                      'physical_samples': samples, 'observations_per_modality': len(identities),
                      'sample_ids': list(fused_samples.sample_ids),
                      'sample_embedding_shape': list(fused_samples.embeddings.shape),
                      'symmetric_info_nce': float(loss.detach()),
                      'gradient_computed': encoder_b.projection.weight.grad is not None,
                      'top1_a_to_b_retrieval': retrieval}, indent=2))


if __name__ == '__main__':
    main()
