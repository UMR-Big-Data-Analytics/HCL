# HCL

Hyperspherical
Concentration-Aware Linkage for Hierarchical Clustering for text embeddings derived from a von-Mises Fisher distribution.
## Installation

The project requires Python 3.14 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

```python
import numpy as np

from hcl.core import HclHAC

# X is an array of shape (n_samples, n_features).
X = np.asarray(...)

clusterer = HclHAC(
    n_clusters=10,
    gamma=0.075,
)
labels = clusterer.fit_predict(X)
```

Input vectors are normalized by default. To vary `gamma` throughout the
hierarchy, pass `progressive=True` with a `(start, end)` tuple:

```python
clusterer = HclHAC(
    n_clusters=10,
    gamma=(0.01, 0.075),
    progressive=True,
)
```

## Experiments

Datasets and embedding models used by the experiments are configured in
`config.json`. Generate embeddings for all configured pairs:

```bash
uv run generate-embeddings
```

Or select one model and dataset:

```bash
uv run generate-embeddings \
  --model intfloat/multilingual-e5-large \
  --dataset mteb/arxiv-clustering-p2p
```

Run all experiments (will take a long time) or plotting suites:

```bash
uv run run-all-experiments
uv run run-all-plots
```

Results are written under `results/`.
