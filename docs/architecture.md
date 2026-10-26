# Architecture and scientific contracts

## Observation versus physical sample

A physical sample is the independent entity, while an observation is a view or
measurement of it. One sample can yield many observations in each modality.
IDs must be truthful and stable across views. Partition membership must be
assigned externally at the physical-sample level before deriving observations.
No filename parsing or latent-vector matching substitutes for sample identity.

## Modality-specific representation

BaseEncoder is a torch.nn.Module with embedding_dim and an abstract forward.
Input structure belongs to the modality; output is [B,D], preserving row identity.
IdentityEncoder validates existing features. LinearEncoder handles vector inputs
and supports ordinary autograd. The package does not download or wrap large
pretrained networks. Subclasses must uphold the finite output and identity contract.

## Shared space and fusion

Equal dimensions alone do not establish semantic alignment. Encoders must share
coordinates by design or training before averaging features. mean_fusion accepts
named [B,D] tensors and sorts names for deterministic summation. Caller-controlled
row identity is a prerequisite. None may omit a whole modality explicitly;
per-sample missingness and implicit zero padding are unsupported.

mean_logits instead pools [B,C] class logits, assuming identical class ordering
and comparable scale. Softmax follows pooling if needed. It is mathematically
different from pooling softmax probabilities and is not a calibrated confidence.

## Sample aggregation

aggregate_samples maps each observation ID to a physical sample via explicit
sample_ids. It returns SampleEmbeddings with both IDs and rows, ordered by first
appearance. The mean is differentiable. Equal observation weights can overweight
images with more crops: average crops per image, then images per sample when
equal-image weighting is intended. Aggregate separately by modality and split.

## Contrastive alignment

For normalized paired vectors a_i,b_i, symmetric InfoNCE is half the sum of the
A-to-B and B-to-A cross-entropies of the cosine matrix divided by temperature.
Diagonal entries are positives; all off-diagonal entries are negatives. Each
batch therefore needs at least two distinct physical samples, one row per
modality/sample. Optional ID arguments verify uniqueness and row pairing. They
do not construct joins or reorder a mismatched table. Without IDs, the caller
must ensure pairing. Same-class samples are still negatives; class-aware masking
and multi-positive objectives are outside this version.

InfoNCE rejects zero vectors and uses epsilon-regularized L2 normalization:
`x / max(||x||_2, eps)`, with `eps = sqrt(torch.finfo(compute_dtype).tiny)`.
This is the threshold below which squared norms leave the dtype's normal range:
approximately 1.0842e-19 for float32 and 1.4917e-154 for float64. Half/bfloat16
inputs are promoted to float32 before choosing the epsilon. The norm floor is
numerical regularization, not coordinate clipping or gradient clipping. Above
it, the ordinary cosine objective and gradients are preserved; below it,
vectors are intentionally not forced to unit length. Exact unit normalization
at arbitrarily small magnitudes has derivatives proportional to 1/||x||, which
cannot always be represented in the input dtype.

For overflow protection, InfoNCE first divides by `max(max(abs(x)), 1)` and
passes an explicit epsilon to torch.nn.functional.normalize. A row scaled this
way has norm at least one, so epsilon is inactive for scaled rows; otherwise
it applies directly to the original row norm. This implements the stated norm
floor while avoiding the previous tiny pre-scaling divisor and its overflowing
backward reciprocal. Retrieval's exact-cosine normalization is unchanged.

Cross-entropy is computed directly on logits, not manual exponentials.
Nonfinite logits/loss at an unrepresentable temperature fail clearly; this
fix does not promise finite gradients for every extreme temperature/dtype
combination or arbitrarily large upstream gradient. Validation preserves
gradients but performs scalar checks, which can synchronize accelerator
execution; CPU is the validated target.

## Evaluation and downstream uses

Cosine matrices support N query and M gallery rows with shared D. Top-1 retrieval
matches explicit physical IDs, not diagonal positions. Tables have unique IDs;
the gallery may have extra distractors but must contain every query sample.
Ties resolve to the first gallery row. A-to-B and B-to-A need separate calls.

Fused or unimodal sample embeddings can feed an externally implemented classifier,
nearest-neighbor search, clustering or acquisition policy. The primitives neither
train such systems nor claim generalization from the constructed example.

## Leakage risks

Independent crop/view splitting leaks specimen-specific information across
partitions. Joining modality tables by row position can silently pair different
specimens. Treating repeated views as InfoNCE negatives creates false negatives.
Treating seed replicates as independent test entities creates pseudoreplication.
Cross-partition pooling also leaks information even without labels. The package
validates what its APIs can observe; split manifests, label consistency, aliases,
train-only preprocessing and appropriate statistical inference remain external
responsibilities. The benchmarking portfolio package is not a runtime dependency.
