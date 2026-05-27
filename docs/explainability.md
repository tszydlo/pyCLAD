# Explainability

### Overview

The `pyclad.explainability` module provides **post-hoc, feature-level interpretability** for
continual anomaly detection experiments. After each concept is learned, it analyses the
strategy's replay buffer through a surrogate pipeline (Isolation Forest - Random Forest) to
identify which features most influence anomaly scores, and tracks how those feature signatures
evolve across concepts (*concept drift*).

The module is organised into three sub-packages:

- **`data_generation`** – Synthetic concept dataset builder with configurable signal features
  for controlled drift experiments.
- **`metrics`** – Callbacks that plug into the pyCLAD scenario lifecycle and compute drift
  metrics after each concept is trained.
- **`visualization`** – Heatmap visualiser of per-concept feature importances.

All metric callbacks implement `pyclad.callbacks.callback.Callback` and can be passed in the
`callbacks` list of any existing pyCLAD scenario class.

---

### data_generation

The `data_generation` sub-package provides a synthetic dataset generator for controlled
concept-drift experiments. It lets you specify exactly which features carry the anomaly
signal in each concept, making it easy to verify that drift-detection callbacks behave as
expected.

#### SyntheticConceptGenerator

`SyntheticConceptGenerator` builds a `ConceptsDataset` from a user-supplied signal-feature
mapping. Each concept is an independent Gaussian cloud; anomaly samples in concept *i*
receive a mean shift of `signal_strength` on the feature indices listed for that concept in
`concept_signals`. Changing which features carry the signal between concepts simulates
concept drift.

##### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `concept_signals` | `dict` | — | `{concept_name: [feature_idx, ...]}` mapping of signal features per concept |
| `n_features` | `int` | `20` | Total number of features per sample |
| `n_train` | `int` | `150` | Training samples per concept |
| `n_test_normal` | `int` | `60` | Normal test samples per concept |
| `n_test_anomaly` | `int` | `40` | Anomalous test samples per concept |
| `signal_strength` | `float` | `5.0` | Mean shift applied to anomaly signal features |
| `seed` | `int` | `42` | Base random seed (each concept uses `seed + offset`) |

##### Key methods

- `build_dataset()` -> `ConceptsDataset` — generate all concepts and return a ready-to-use dataset.
- `export_csv(path)` — save all data (train + test, all concepts) to a single CSV file.
- `export_splits(train_path, test_path)` — save separate train/test CSV files.

##### Properties

- `feature_names` — list of feature names `['f0', 'f1', ..., 'fN-1']`.
- `concept_order` — concept names in insertion order.
- `concept_signals` — the signal-feature mapping passed at construction.

### Code Example

    concept_signals = {
        "C0": [10, 11],   # concepts differ -> drift expected
        "C1": [12, 13],
        "C2": [14, 15],
    }
    gen = SyntheticConceptGenerator(concept_signals=concept_signals, n_features=20, seed=42)
    dataset = gen.build_dataset()
    gen.export_csv("dataset.csv")

---

### metrics

The `metrics` sub-package provides three callbacks that measure feature-level concept drift
by inspecting the strategy's replay buffer after each concept is learned.

**Callback order matters**: `ModelComparisonCallback` must appear **before**
`DriftMetricsCallback` and `WassersteinDriftCallback` in the `callbacks` list, because
both read `rf_importances` that is populated by `ModelComparisonCallback` during the same
`after_training` hook.

#### ModelComparisonCallback

Scores the replay buffer with **Isolation Forest**, then trains a **Random Forest** to
extract feature importances as a proxy for concept-level anomaly signals.

The surrogate pipeline runs `after_training` for every concept:

1. Fit an `IsolationForestAdapter` on the current replay buffer (contamination = 10 %).
2. Assign pseudo-label `1` (anomaly) to the top 10 % by IF score, `0` to the rest.
3. If two or more distinct pseudo-labels exist, fit a `RandomForestClassifier` (200 trees)
   on the buffer using those pseudo-labels.
4. Store the RF `feature_importances_` array in `rf_importances[concept_name]`.

The importances stored in `rf_importances` are consumed by `DriftMetricsCallback` and
`WassersteinDriftCallback`. A concept entry is absent from `rf_importances` if the buffer's
pseudo-label array is constant (edge case: the entire buffer is classified as either all-normal
or all-anomaly).

##### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `strategy` | `ReplayEnhancedStrategy` | — | Strategy whose buffer is read after each concept |
| `feature_names` | `list[str]` | — | Feature-name strings used for the Top-2 importance printout |
| `seed` | `int` | `42` | Random seed for both Isolation Forest and Random Forest |

##### Public attributes

- `rf_importances` (`dict[str, ndarray]`) — maps each concept name to a 1-D array of RF
  feature importances (length = number of features).

### Code Example

    mc_cb = ModelComparisonCallback(strategy, feature_names, seed=42)

---

#### DriftMetricsCallback

Computes **cosine similarity** and **soft top-$k$ churn** between consecutive concept
feature-importance vectors to detect feature-level concept drift.

**Cosine similarity** measures the directional alignment of importance vectors:

$$\text{cos\_sim} = \frac{\mathbf{f}_{i-1} \cdot \mathbf{f}_i}{\|\mathbf{f}_{i-1}\| \, \|\mathbf{f}_i\|}$$

A value close to 1 indicates that the set of important features is stable (no drift); a
lower value indicates that importance has shifted to different features (drift).

**Soft top-$k$ churn** measures magnitude-weighted feature turnover:

$$\text{churn} = \frac{|\text{top}_k(\mathbf{f}_{i-1}) \,\triangle\, \text{top}_k(\mathbf{f}_i)|}{2k}
               \times \sum_j |f_{i-1,j} - f_{i,j}|$$

Both metrics are computed from the **second concept** onwards (nothing to compare against
for the first concept).

A ground-truth `expected_drift` label is also stored for benchmarking, derived from the
overlap between the `concept_signals` entries of the previous and current concept:

| Signal-feature overlap | `expected_drift` |
|---|---|
| No shared signal features | `"HIGH"` |
| Exactly one shared signal feature | `"MODERATE"` |
| All signal features shared | `"LOW"` |

##### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mc_callback` | `ModelComparisonCallback` | — | Source of `rf_importances` |
| `concept_signals` | `dict` | — | Ground-truth signal-feature map for `expected_drift` |
| `k` | `int` | `5` | Number of top features used for hard-churn calculation |

##### Public attributes

- `results` (`dict[str, dict]`) — keyed by concept name (from the second concept onwards).
  Each entry contains:
  - `cos_sim` (*float*) — cosine similarity in `[-1, 1]`.
  - `churn` (*float*) — soft top-*k* churn value (non-negative).
  - `expected_drift` (*str*) — `'HIGH'`, `'MODERATE'`, or `'LOW'`.

### Code Example

    drift_cb = DriftMetricsCallback(mc_cb, concept_signals, k=5)

---

#### WassersteinDriftCallback

Measures **per-feature Earth Mover's Distance (Wasserstein-1 distance)** between the
replay buffer contents at consecutive concept boundaries.

The callback operates in two modes:

- **Raw mode** (`mc_callback=None`, default) — computes only the per-feature EMD between
  the previous and current buffer snapshot. This is fully backward-compatible and produces
  identical results across all model-swap experiments because the buffer contents are
  scorer-independent.
- **Weighted mode** (`mc_callback=<ModelComparisonCallback>`) — additionally computes an
  importance-weighted Wasserstein score:

$$\text{weighted\_wasserstein} = \sum_{f=0}^{F-1} \text{importance}[f] \times \text{EMD}(P_f, Q_f)$$

Because the importance weights (`rf_importances`) differ per scorer/extractor combination,
`weighted_wasserstein` is sensitive to the model-swap choice even though the raw buffer EMD
is not. This makes it the key discriminating metric in model-swap studies.

##### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `strategy` | `ReplayEnhancedStrategy` | — | Strategy whose buffer is read after each concept |
| `feature_names` | `list[str]` | — | Feature-name strings for the `per_feature` dict keys |
| `mc_callback` | `ModelComparisonCallback \| None` | `None` | If provided, enables weighted mode |

##### Public attributes

- `results` (`dict[str, dict]`) — keyed by concept name (from the second concept onwards).
  Each entry always contains:
  - `mean_wasserstein` (*float*) — mean EMD across all features.
  - `max_wasserstein` (*float*) — max EMD across all features.
  - `per_feature` (*dict*) — `{feature_name: raw_emd}` mapping.

  When `mc_callback` is provided, each entry also contains:
  - `weighted_wasserstein` (*float*) — importance-weighted EMD score.

### Code Example

    # Raw mode (backward-compatible):
    wass_cb = WassersteinDriftCallback(strategy, feature_names)

    # Weighted mode (sensitive to scorer/extractor choice):
    wass_cb = WassersteinDriftCallback(strategy, feature_names, mc_callback=mc_cb)

    mc_cb    = ModelComparisonCallback(strategy, feature_names, seed=42)
    drift_cb = DriftMetricsCallback(mc_cb, concept_signals, k=5)
    wass_cb  = WassersteinDriftCallback(strategy, feature_names, mc_callback=mc_cb)
    roc_cb   = ConceptMetricCallback(RocAuc(), [ContinualAverage(), BackwardTransfer(), ForwardTransfer()])

    scenario = ConceptIncrementalScenario(
        dataset, strategy=strategy,
        callbacks=[roc_cb, mc_cb, drift_cb, wass_cb],
    )
    scenario.run()

---

### visualization

The `visualization` sub-package provides a heatmap visualiser that displays per-concept
Random Forest feature importances. It makes it easy to see at a glance which features were
most important in each concept and how importance shifted across concepts.

#### FeatureImportanceHeatmap

Renders a **seaborn heatmap** of per-concept RF feature importances with the following layout:

- **Y-axis** — features (`f0` at top, `fN-1` at bottom). Features that appear in any
  concept's ground-truth signal set are annotated with an ` X` suffix.
- **X-axis** — concepts in the order supplied by `concept_order`.
- **Cell colour** — RF importance value on the `YlOrRd` palette, anchored at 0.
- **Cell text** — rounded importance value (3 decimal places).

##### Constructor parameters

| Parameter | Type | Description |
|---|---|---|
| `concept_order` | `list[str]` | Ordered concept names for the x-axis |
| `concept_signals` | `dict` | Ground-truth signal map used to annotate signal features with ` X` |
| `feature_names` | `list[str]` | Feature name strings, e.g. `['f0', 'f1', ...]` |
| `n_features` | `int` | Total number of features (y-axis rows) |

##### `plot()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `importances_dict` | `dict[str, ndarray]` | — | Per-concept importance arrays; absent concepts shown as all-zero rows |
| `scenario_name` | `str` | — | Label displayed in the figure title |
| `n_top` | `int` | `5` | Retained for API compatibility — all features are shown |
| `save_path` | `str \| None` | `None` | Path to save figure (PNG or PDF at 150 dpi) |
| `show` | `bool` | `True` | Whether to call `plt.show()` after rendering |
| `prefix` | `str` | `""` | Filename prefix — retained for API compatibility, currently unused |

### Code Example

    hm = FeatureImportanceHeatmap(
        concept_order=gen.concept_order,
        concept_signals=concept_signals,
        feature_names=gen.feature_names,
        n_features=20,
    )
    hm.plot(mc_cb.rf_importances, "MyExperiment", save_path="heatmap.png", show=False)

---

### Full Code Example

The following self-contained example mirrors the Explainability Drift Example
from the repository. It runs two Concept-Incremental scenarios side by side one with
rotating signal features (drift) and one with stable signal features (no drift) and
produces a comparison table and two heatmaps.

```python linenums="1"
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.adapters.pyod_adapters import LocalOutlierFactorAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.replay.buffers.adaptive_balanced import AdaptiveBalancedReplayBuffer
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy
from pyclad.strategies.replay.selection.random import RandomSelection
from pyclad.explainability.data_generation.concept_generator import SyntheticConceptGenerator
from pyclad.explainability.metrics.drift_metrics_callback import DriftMetricsCallback
from pyclad.explainability.metrics.feature_importance_callback import ModelComparisonCallback
from pyclad.explainability.metrics.wasserstein_callback import WassersteinDriftCallback
from pyclad.explainability.visualization.heatmaps import FeatureImportanceHeatmap

SEED        = 42
N_FEATURES  = 20
FEATURE_NAMES = [f"f{i}" for i in range(N_FEATURES)]

# DRIFT: signal features rotate every concept
DRIFT_SIGNALS = {"C0": [10, 11], "C1": [12, 13], "C2": [14, 15], "C3": [16, 17], "C4": [18, 19]}

# NO-DRIFT: same signal features every concept
NODRIFT_SIGNALS = {"C0": [10, 11], "C1": [10, 11], "C2": [10, 11], "C3": [10, 11], "C4": [10, 11]}


def run_experiment(concept_signals, label):
    random.seed(SEED)
    np.random.seed(SEED)

    gen      = SyntheticConceptGenerator(concept_signals=concept_signals, n_features=N_FEATURES, seed=SEED)
    dataset  = gen.build_dataset()

    model    = LocalOutlierFactorAdapter(n_neighbors=20, contamination=0.1)
    buffer   = AdaptiveBalancedReplayBuffer(selection_method=RandomSelection(), max_size=200)
    strategy = ReplayEnhancedStrategy(model=model, buffer=buffer)

    roc_cb   = ConceptMetricCallback(RocAuc(), [ContinualAverage(), BackwardTransfer(), ForwardTransfer()])
    mc_cb    = ModelComparisonCallback(strategy, FEATURE_NAMES, seed=SEED)
    drift_cb = DriftMetricsCallback(mc_cb, concept_signals, k=5)
    wass_cb  = WassersteinDriftCallback(strategy, FEATURE_NAMES, mc_callback=mc_cb)

    ConceptIncrementalScenario(
        dataset=dataset, strategy=strategy,
        callbacks=[roc_cb, mc_cb, drift_cb, wass_cb],
    ).run()

    JsonOutputWriter(pathlib.Path(f"explainability_{label}_output.json")).write(
        [model, dataset, strategy, roc_cb]
    )
    return gen, mc_cb, drift_cb, wass_cb, list(concept_signals.keys())


gen_d, mc_d, drift_d, wass_d, order_d = run_experiment(DRIFT_SIGNALS,   "drift")
gen_n, mc_n, drift_n, wass_n, order_n = run_experiment(NODRIFT_SIGNALS, "nodrift")

for label, drift_cb, wass_cb, order in [
    ("DRIFT",    drift_d, wass_d, order_d),
    ("NO-DRIFT", drift_n, wass_n, order_n),
]:
    print(f"\n{'='*90}\n  {label} - Drift Metrics\n{'='*90}")
    print(f"{'Transition':<12} {'GT':>8} {'CosSim':>10} {'Churn':>10} {'Raw Wass':>12} {'Wtd Wass':>12}")
    for cname in [c for c in order[1:] if c in drift_cb.results]:
        prev = order[order.index(cname) - 1]
        r, w = drift_cb.results[cname], wass_cb.results.get(cname, {})
        print(
            f"  {prev}->{cname:<5} {r['expected_drift']:>8} "
            f"{r['cos_sim']:>10.3f} {r['churn']:>10.3f} "
            f"{w.get('mean_wasserstein', float('nan')):>12.4f} "
            f"{w.get('weighted_wasserstein', float('nan')):>12.4f}"
        )

for gen, mc_cb, signals, name in [
    (gen_d, mc_d, DRIFT_SIGNALS,   "drift"),
    (gen_n, mc_n, NODRIFT_SIGNALS, "nodrift"),
]:
    FeatureImportanceHeatmap(gen.concept_order, signals, FEATURE_NAMES, N_FEATURES).plot(
        mc_cb.rf_importances, name.capitalize(), save_path=f"explainability_{name}_heatmap.png", show=False
    )

plt.show()
```

---

### Optional Dependencies

The `pyclad.explainability` module requires `scipy` (for Wasserstein distance computation via
`scipy.stats.wasserstein_distance`). It is not included in the default pyCLAD installation.

Install the full explainability extras with:

    pip install pyclad[explainability]

or install only the required packages separately:

    pip install scipy>=1.7 scikit-learn>=1.0 seaborn>=0.12
