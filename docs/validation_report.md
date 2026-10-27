# Validation report

Version: 0.1.0. Latest validation date: 2026-10-27.

## Environment and executed checks

- Python: 3.11.15; PyTorch 2.11.0+cu128; NumPy 2.4.4; pytest 9.1.1.
- Execution: CPU only, CUDA hidden during validation. No network is used by tests
  or the demo. Dependency installation can require a package index.
- A repository-local virtual environment was created with
  `python -m venv --system-site-packages .venv`. It reuses the existing ML
  dependencies read-only; the external environment was not modified. This
  validation is not a claim of testing a wholly fresh dependency stack.
- `python -m pip install -e ".[dev]" --no-build-isolation --no-index --disable-pip-version-check`
  passed in the existing local environment; all dependencies were already present.
  Build caches and temporary files were confined to ignored local paths.
- `python -m compileall src tests examples`: passed, exit 0. Bytecode was
  redirected into ignored local validation storage.
- `pytest -q`: **60 passed, 0 failed, 0 skipped**. Tests cover encoders, embedding
  and mean-logit fusion, InfoNCE/autograd/pair identity, sample means, ID-aware
  retrieval, invalid inputs and distribution/package version consistency.
- `python examples/synthetic_multimodal_demo.py`: passed after the stability fix. Six artificial physical samples
  each have three observations per modality; the fused sample table is [6,8].
  Gradients were computed. The known constructed alignment yielded InfoNCE
  0.00022848813387099653 and A-to-B top-1 retrieval 1.0. These are synthetic
  mechanics checks, not trained or held-out scientific performance.
- A Python version assertion checked project metadata, installed distribution
  metadata and `multimodal_repr.__version__`: all equal **0.1.0**.
- One warning originates in the inherited PyTorch environment's deprecated
  pynvml installation. It does not indicate a test failure or a runtime GPU
  requirement. No external packages were changed to suppress it.

## Public-content audit

Public source, tests, docs and configuration were scanned for Windows drive
paths, user-home paths, local file URLs and credential assignments: none found.
No actual dataset, model checkpoint, secret or token is included. All fixture
identities and tensors are artificial. Ignored environment/log/build files can
contain local installation paths; they are excluded from public files and Git.
No files outside this new repository were written. The source scientific project
and the existing benchmarking repository were untouched.

## Scientific limitations

- This is a primitive library, not a trainer or a reproduction of historical
  experiments. No pretrained encoders, learned material tokens or downstream
  classifier are supplied.
- Modality fusion requires shared semantics and paired row order; it cannot infer
  identity or align arbitrary independently trained spaces.
- InfoNCE treats other physical samples as negatives, even within the same class.
  Aggregate duplicate views first; optional IDs enforce one paired sample per row.
- Sample aggregation has no split/label metadata and must be called separately
  per modality and partition. A separate immutable manifest is needed to enforce
  sample-level split and label consistency.
- Optional missingness is whole-modality only. Retrieval is directional and ties
  select the first gallery entry. Zero vectors have undefined cosine and fail.
- Determinism is tested for this CPU stack, not all hardware/library versions.

## Final public repository tree

Local environments, validation logs, generated metadata and Git internals are
excluded from this public file inventory.

```text
.github/workflows/tests.yml
.gitignore
LICENSE
README.md
assets/README.md
configs/example.yaml
docs/architecture.md
docs/validation_report.md
examples/synthetic_multimodal_demo.py
pyproject.toml
src/multimodal_repr/__init__.py
src/multimodal_repr/aggregation/__init__.py
src/multimodal_repr/aggregation/sample.py
src/multimodal_repr/encoders/__init__.py
src/multimodal_repr/encoders/base.py
src/multimodal_repr/evaluation/__init__.py
src/multimodal_repr/evaluation/retrieval.py
src/multimodal_repr/fusion/__init__.py
src/multimodal_repr/fusion/late.py
src/multimodal_repr/fusion/mean.py
src/multimodal_repr/losses/__init__.py
src/multimodal_repr/losses/contrastive.py
src/multimodal_repr/validation/__init__.py
src/multimodal_repr/validation/inputs.py
tests/test_aggregation.py
tests/test_contrastive.py
tests/test_encoders.py
tests/test_fusion.py
tests/test_retrieval.py
```

## Local Git handoff

This validation concerns a new follow-up commit; the original initial commit
5b11a0a is preserved. `git diff`, `git diff --check` and `git status` were inspected
before staging. The final response records the new commit hash and final status.
No amendment, remote, push or release is authorized or performed. This report
cannot contain its own commit hash without changing that hash.


## Near-zero InfoNCE stability correction

### Reproduced root cause

The original normalization divided each row by max(abs(x)), including scales
around 1e-40. Forward ratios were finite, but their backward reciprocal is on
the order of 1e40, beyond float32's maximum finite magnitude (about 3.4e38).
Autograd anomaly detection located `DivBackward0` at that division; both input
gradients became NaN. The original reproduction had finite forward loss
0.05034204199910164 but nonfinite gradients. Exact unit normalization has a
Jacobian proportional to 1/||x||, so finite float32 gradients for arbitrarily
tiny vectors require an explicit regularization convention.

### Minimal fix and scientific behavior

Only InfoNCE uses a new private normalization helper. It computes
`x / max(||x||_2, eps)` with an explicit dtype-aware norm floor
`eps = sqrt(torch.finfo(compute_dtype).tiny)`, approximately 1.0842e-19 for
float32. Half/bfloat16 are promoted to float32 first. This threshold is tied
to squared-norm underflow, rather than an arbitrary default constant.

Before F.normalize, rows are divided by `max(max(abs(x)), 1)`. This protects
large inputs without ever dividing by a subnormal pre-scaling factor. Scaled
rows have norm at least one so the epsilon is inactive there; unscaled rows
use the stated floor directly. No coordinates or gradients are clipped.

Above the norm floor the cosine objective and its derivatives are unchanged
up to floating-point roundoff. Below it normalization is intentionally
regularized; vectors need not have unit length. Pairing, temperature and the
symmetric A-to-B/B-to-A cross-entropies are unchanged. Exactly zero vectors
still fail. All shape, dtype/device, finite-value and ID checks remain intact.
Retrieval's exact-cosine helper and all other modules are unchanged. No new
dependency or package-version change was introduced.

### New regression coverage and actual results

The suite grew from 48 to 60 tests. Twelve new cases cover float32 scales
1e-40, 1e-30, 1e-10, 1 and 1e30; float64 scale 1e-200; an explicit norm-floor
forward/gradient reference; ordinary float32/float64 forward/gradient agreement
with unregularized cosine; float64 finite-difference gradcheck; and unchanged
zero-row rejection in either modality. Existing aligned-versus-shuffled,
shape, identity, temperature and other tests also pass.

- Compilation: exit 0.
- Offline editable reinstall with the dev extra: exit 0.
- pytest: 60 passed, 0 failed, 0 skipped (one pre-existing environment warning).
- Synthetic CPU demo: exit 0; same constructed loss and retrieval output as the
  original ordinary-magnitude example.
- Version assertion: package, distribution and project all remain 0.1.0.
- Explicit standalone float32 near-zero regression: exit 0; finite forward
  loss 1.0986123085021973 and finite, nonzero gradients in both modalities.
  Maximum absolute gradient in each modality: 0.009452237747609615.
- Public tracked-file scan: no private absolute paths or credentials detected.
- Only the loss, its tests, README, architecture and this report were changed.

The explicit standalone regression that was executed was:

```python
import torch
from multimodal_repr.losses.contrastive import symmetric_info_nce

a = ((torch.eye(3, dtype=torch.float32) + 0.1) * 1e-40).requires_grad_()
b = ((torch.eye(3, dtype=torch.float32) + 0.2) * 1e-40).requires_grad_()
loss = symmetric_info_nce(a, b, temperature=0.2)
assert torch.isfinite(loss).item()
loss.backward()
for value in (a, b):
    assert value.grad is not None
    assert torch.isfinite(value.grad).all().item()
```

The numerical guarantee is targeted: it does not promise finite derivatives
for every extreme temperature/dtype combination or arbitrarily large upstream
gradient. All tests and measurements above used CPU synthetic inputs only.


## GitHub Actions CI for v0.1.0

Validation date: 2026-10-27. Workflow: `.github/workflows/tests.yml`.

### Workflow scope

- Triggers: pushes to `main` and pull requests targeting `main`.
- Runner: `ubuntu-latest`; Python **3.12**; CPU execution only.
- Official actions: `actions/checkout@v7` and `actions/setup-python@v7`.
  Stable major versions were checked against the official
  [checkout release](https://github.com/actions/checkout/releases/tag/v7.0.1)
  and [setup-python release](https://github.com/actions/setup-python/releases/tag/v7.1.0).
- Named steps check out code without retaining credentials, set up Python,
  upgrade pip, install CPU PyTorch, and install the project with its dev extra
  using `python -m pip install -e ".[dev]"`.
- Compilation: `python -m compileall src tests examples`.
- Full existing 60-test suite: `pytest -q`.
- Synthetic demo: `python examples/synthetic_multimodal_demo.py`.
- Package import/version check verifies project metadata, installed distribution
  metadata and `multimodal_repr.__version__` all equal **0.1.0**.
- Near-zero smoke check calls the public symmetric_info_nce API with paired
  float32 embeddings around 1e-40 and requires_grad=True. It asserts a finite
  loss, calls backward(), and asserts both gradients exist and contain neither
  NaN nor Inf. It fails the job on any invalid result.
- No dataset, GPU, repository secret or post-install application network access
  is required. Only dependency installation needs package downloads. The job
  uses read-only repository permissions; there are no publishing/release steps.
- README includes the native GitHub Actions badge for this exact workflow and
  repository. The badge is dynamic, not a claim of a completed hosted run.

### Structural and local-equivalent validation

**GitHub-hosted CI has not run for this new workflow.** Its real Python 3.12 /
Ubuntu execution must occur after push. All results below are local-equivalent
checks on the existing Python **3.11.15** environment, not a hosted-CI success
claim. Existing ML dependencies were reused; no CPU-wheel replacement or pip
upgrade was needed locally.

- Complete YAML was parsed, with duplicate mapping keys rejected. Main-only
  event branches, Ubuntu runner, Python version, permissions, official action
  majors and all ten named steps were checked. Indentation contains no tabs.
  Both embedded Python blocks compiled and ran exactly as stored in the YAML.
- Compilation passed, exit 0.
- Editable installation with the dev extra passed, exit 0. Local flags
  `--no-build-isolation --no-index --disable-pip-version-check` reused existing
  dependencies without package-index access. Installation changed only the
  existing repository-local virtual environment.
- Full suite: **60 passed, 0 failed, 0 skipped**. The inherited environment's
  pre-existing pynvml warning remains nonfatal.
- Synthetic demo passed, exit 0. Constructed example loss remains
  0.00022848813387099653; A-to-B retrieval remains 1.0. These are synthetic
  mechanics checks, not scientific generalization claims.
- Exact workflow version/import block passed: **0.1.0** everywhere.
- Exact workflow near-zero smoke passed: loss **1.0986123085**, gradients finite
  in both modalities. Six injected failure controls confirmed the guard rejects
  NaN/Inf forward losses and NaN/Inf gradients in either modality.
- Scientific package code, existing tests, example and pyproject were verified
  unchanged with `git diff --exit-code` restricted to those paths.
- Public file scan found no machine-specific paths, private information or
  credential assignments. Git diff/status were inspected before the new commit.

### Git handoff

Only the workflow, README badge and this report changed. The existing origin
remote is retained unchanged. The new commit is separate from the numerical
stability commit; no existing commit is amended. No push, tag or release is
performed. The final response reports the resulting local commit and status.
