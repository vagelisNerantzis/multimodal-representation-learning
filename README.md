# Multimodal Representation Learning

A small PyTorch package for aligned multimodal embeddings, fusion and
physical-sample-level representation evaluation. Version **0.1.0**.

## Scientific motivation

Conventional image-level ML often treats observations independently. Scientific
datasets frequently contain several images, spectra or views of the same
physical specimen. Those observations share identity: they are neither independent
train/test samples nor necessarily valid contrastive negatives for one another.

This framework is domain-independent. Textile imaging across mobile, macro and
microscopy views is one motivating example, not a hard-coded data model. No
private dataset, pretrained model, historical metric or performance claim is included.

## Core abstraction

```text
physical sample -> multimodal observations -> modality-specific encoders
    -> aligned/shared representation -> fusion -> sample-level inference
```

Sample IDs remain explicit during aggregation and retrieval. Observations must
inherit their physical sample's partition; this package never creates a split.

## Key features

- PyTorch encoder interface with identity and trainable linear implementations.
- Deterministic equal-weight embedding fusion and mean-logit late fusion.
- Symmetric, internally normalized InfoNCE with optional paired-ID verification.
- Differentiable mean aggregation with first-appearance sample ordering.
- Cosine matrices and ID-aware cross-modal top-1 retrieval independent of row order.
- Validation of shape, dtype, device, finite values, IDs and temperature.
- CPU-only, deterministic synthetic tests; no network or notebook requirement.

## Installation

Python 3.10+ is required. Choose a suitable PyTorch build for your machine if
needed; no CUDA-specific wheel is pinned.

```bash
python -m pip install -e ".[dev]"
```

Runtime dependencies are torch and numpy. pytest is a development extra. YAML
is illustrative configuration, not a runtime dependency.

## Minimal usage

```python
import torch
from multimodal_repr.encoders.base import IdentityEncoder
from multimodal_repr.aggregation.sample import aggregate_samples
from multimodal_repr.fusion.mean import mean_fusion
from multimodal_repr.losses.contrastive import symmetric_info_nce
from multimodal_repr.evaluation.retrieval import top1_cross_modal_accuracy

ids = ["sample_a", "sample_a", "sample_b", "sample_b"]
a = torch.tensor([[1., 0.], [0.9, 0.1], [0., 1.], [0.1, 0.9]])
b = a.clone()
encoded = IdentityEncoder(2)(a)
sa = aggregate_samples(encoded, ids)
sb = aggregate_samples(b, ids)
loss = symmetric_info_nce(sa.embeddings, sb.embeddings,
                         sample_ids_a=sa.sample_ids, sample_ids_b=sb.sample_ids)
fused = mean_fusion({"sensor_a": sa.embeddings, "sensor_b": sb.embeddings})
accuracy = top1_cross_modal_accuracy(sa.embeddings, sb.embeddings,
                                     sa.sample_ids, sb.sample_ids)
```

This constructed example checks mechanics, not empirical generalization.
Fusion assumes common row identity and shared coordinates. It does not align
separately trained encoders automatically. `mean_logits` averages logits with a
common class order; it does not average probabilities.

## Synthetic demo

```bash
python examples/synthetic_multimodal_demo.py
```

The demo uses six artificial samples, three observations per modality and two
linear encoders with deliberately known alignment. It computes a differentiable
contrastive loss, fuses observations, aggregates to samples and evaluates retrieval.
High retrieval in this constructed setting is not evidence of learned performance.
Equivalent illustrative settings are in `configs/example.yaml`; the example
accepts `--seed`, `--embedding-dim` and `--temperature` without a YAML loader.

## Testing

```bash
python -m compileall src tests examples
python -m pytest -q
```

All tests use synthetic CPU tensors and require no downloads or GPU.

## Repository structure

```text
src/multimodal_repr/
  encoders/base.py           encoder contract, identity and linear encoders
  fusion/mean.py             embedding-level fusion
  fusion/late.py             mean-logit prediction fusion
  losses/contrastive.py      symmetric InfoNCE
  aggregation/sample.py     explicit observation-to-sample means
  evaluation/retrieval.py   cosine and ID-aware top-1 retrieval
  validation/inputs.py      shared tensor and identity checks
tests/                      synthetic pytest suite
examples/                   fast CPU demonstration
configs/example.yaml        illustrative parameters
docs/architecture.md        assumptions and scientific contracts
docs/validation_report.md   executed checks and full file tree
assets/README.md            public asset policy
```

## Reproducibility principles

Split at the physical-sample level before producing crops, embeddings or modality
tables. Keep all views of each specimen in the same partition. Fit encoders and
preprocessing on training samples; choose development settings on validation,
not test outcomes. Preserve sample IDs through every join. Record seeds,
preprocessing, encoder versions, pooling weights and retrieval direction.

The library does not mutate global seeds. The standalone demo explicitly seeds
its own CPU experiment. Stable modality-name sorting fixes summation order;
bitwise equivalence across hardware, dtypes or PyTorch versions is not promised.

## Scope and current limitations

Version 0.1.0 provides auditable building blocks, not a complete trainer. It has
no large pretrained encoders, model zoo, data loader, optimizer loop, learned
class/material tokens, downstream classifier, calibration or dataset split manager.
Consumers can build such components around the encoder interface.

Mean fusion requires a shared dimension, coordinate system and row order.
Whole modalities may be omitted as None with allow_missing=True; per-row masks
are unsupported. No implicit modality imputation or projection is performed.

InfoNCE uses off-diagonal samples as negatives, including same-class samples.
Unique paired IDs are checked when supplied; otherwise physical pairing is the
caller's responsibility. Aggregate repeated observations before computing loss
or implement a justified multi-positive objective separately.

Aggregation cannot detect cross-partition mixing without partition metadata:
call separately by split/modality. Raw observation means weight samples equally
only after grouping; unequal crops per image require hierarchical aggregation.
Retrieval is directional, requires every query ID in the gallery, and breaks
similarity ties by first gallery position. Cosine evaluation rejects zero vectors.

No real dataset, private checkpoint, scientific benchmark result or state-of-the-art
claim is included. See [architecture](docs/architecture.md) for precise contracts.

## License

[MIT](LICENSE) applies to the software; no dataset or model artifact rights are granted.
