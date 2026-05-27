import logging
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
from pyclad.strategies.replay.buffers.adaptive_balanced import (
    AdaptiveBalancedReplayBuffer,
)
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy
from pyclad.strategies.replay.selection.random import RandomSelection

from pyclad.explainability.data_generation.concept_generator import SyntheticConceptGenerator
from pyclad.explainability.metrics.drift_metrics_callback import DriftMetricsCallback
from pyclad.explainability.metrics.feature_importance_callback import ModelComparisonCallback
from pyclad.explainability.metrics.wasserstein_callback import WassersteinDriftCallback
from pyclad.explainability.visualization.heatmaps import FeatureImportanceHeatmap

logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

SEED = 42
N_FEATURES = 20
N_TRAIN = 150
N_TEST_NORMAL = 60
N_TEST_ANOMALY = 40
SIGNAL_STRENGTH = 5.0
BUFFER_SIZE = 200
FEATURE_NAMES = [f"f{i}" for i in range(N_FEATURES)]

# Signal features ROTATE every concept — clear concept drift
DRIFT_SIGNALS = {
    "C0": [10, 11],
    "C1": [12, 13],
    "C2": [14, 15],
    "C3": [16, 17],
    "C4": [18, 19],
}

# Same signal features every concept — no drift
NODRIFT_SIGNALS = {
    "C0": [10, 11],
    "C1": [10, 11],
    "C2": [10, 11],
    "C3": [10, 11],
    "C4": [10, 11],
}


def _run_experiment(concept_signals, label):
    """Build dataset, run Concept-Incremental scenario, return callbacks and concept order."""
    random.seed(SEED)
    np.random.seed(SEED)

    # Dataset
    gen = SyntheticConceptGenerator(
        concept_signals=concept_signals,
        n_features=N_FEATURES,
        n_train=N_TRAIN,
        n_test_normal=N_TEST_NORMAL,
        n_test_anomaly=N_TEST_ANOMALY,
        signal_strength=SIGNAL_STRENGTH,
        seed=SEED,
    )
    dataset = gen.build_dataset()

    # Model, buffer, strategy
    model = LocalOutlierFactorAdapter(n_neighbors=20, contamination=0.1)
    buffer = AdaptiveBalancedReplayBuffer(selection_method=RandomSelection(), max_size=BUFFER_SIZE)
    strategy = ReplayEnhancedStrategy(model=model, buffer=buffer)

    # Callbacks (order matters: mc_cb must come before wass_cb)
    roc_cb = ConceptMetricCallback(
        base_metric=RocAuc(),
        metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
    )
    mc_cb = ModelComparisonCallback(strategy, FEATURE_NAMES, seed=SEED)
    drift_cb = DriftMetricsCallback(mc_cb, concept_signals, k=5)
    # Pass mc_callback so weighted_wasserstein is also computed
    wass_cb = WassersteinDriftCallback(strategy, FEATURE_NAMES, mc_callback=mc_cb)

    # Run scenario
    scenario = ConceptIncrementalScenario(
        dataset=dataset,
        strategy=strategy,
        callbacks=[roc_cb, mc_cb, drift_cb, wass_cb],
    )
    scenario.run()

    # Save JSON results
    JsonOutputWriter(pathlib.Path(f"explainability_{label}_output.json")).write(
        [model, dataset, strategy, roc_cb]
    )

    concept_order = list(concept_signals.keys())
    return gen, mc_cb, drift_cb, wass_cb, roc_cb, concept_order


if __name__ == "__main__":
    """
    This example shows how to use pyclad.explainability callbacks to track feature-level
    concept drift in a Concept-Incremental scenario with a Replay strategy.
    
    We compare two scenarios side-by-side:
    - DRIFT: signal features rotate every concept (clear concept drift).
    - NO-DRIFT: same signal features are used across all concepts (no concept drift).
    
    We print tables comparing drift detection metrics (cosine similarity, soft churn, raw
    Wasserstein distance, and importance-weighted Wasserstein distance) for each transition,
    and generate heatmaps of the feature importances over time.
    """
    print("=" * 70)
    print("  EXPERIMENT 1: DRIFT  (signal features rotate every concept)")
    print("=" * 70)
    gen_d, mc_d, drift_d, wass_d, roc_d, order_d = _run_experiment(DRIFT_SIGNALS, "drift")
    ll_d = roc_d.info()["concept_metric_callback_ROC-AUC"]["metrics"]
    print(f"  ROC-AUC - Avg={ll_d['ContinualAverage']:.4f}  BWT={ll_d['BackwardTransfer']:.4f}  FWT={ll_d['ForwardTransfer']:.4f}")

    print("=" * 70)
    print("  EXPERIMENT 2: NO-DRIFT  (same signal features every concept)")
    print("=" * 70)
    gen_n, mc_n, drift_n, wass_n, roc_n, order_n = _run_experiment(NODRIFT_SIGNALS, "nodrift")
    ll_n = roc_n.info()["concept_metric_callback_ROC-AUC"]["metrics"]
    print(f"  ROC-AUC - Avg={ll_n['ContinualAverage']:.4f}  BWT={ll_n['BackwardTransfer']:.4f}  FWT={ll_n['ForwardTransfer']:.4f}")

    # Print Drift Metric Tables
    for exp_label, drift_cb, wass_cb, order in [
        ("DRIFT", drift_d, wass_d, order_d),
        ("NO-DRIFT", drift_n, wass_n, order_n),
    ]:
        print(f"\n{'='*90}")
        print(f"  {exp_label} - Drift Metrics (Concept-Incremental)")
        print(f"{'='*90}")
        print(f"{'Transition':<12} {'GT':>8} {'CosSim':>10} {'Churn':>10} {'Raw Wass':>12} {'Wtd Wass':>12}")
        print("-" * 70)
        for cname in [c for c in order[1:] if c in drift_cb.results]:
            prev = order[order.index(cname) - 1]
            r = drift_cb.results[cname]
            w = wass_cb.results.get(cname, {})
            print(
                f"  {prev}->{cname:<5} {r['expected_drift']:>8} "
                f"{r['cos_sim']:>10.3f} {r['churn']:>10.3f} "
                f"{w.get('mean_wasserstein', float('nan')):>12.4f} "
                f"{w.get('weighted_wasserstein', float('nan')):>12.4f}"
            )

    # Generate heatmaps
    print("\nGenerating heatmaps...")

    FeatureImportanceHeatmap(
        concept_order=order_d,
        concept_signals=DRIFT_SIGNALS,
        feature_names=FEATURE_NAMES,
        n_features=N_FEATURES,
    ).plot(
        mc_d.rf_importances,
        "Drift",
        save_path="explainability_drift_heatmap.png",
        show=False,
    )

    FeatureImportanceHeatmap(
        concept_order=order_n,
        concept_signals=NODRIFT_SIGNALS,
        feature_names=FEATURE_NAMES,
        n_features=N_FEATURES,
    ).plot(
        mc_n.rf_importances,
        "No-Drift",
        save_path="explainability_nodrift_heatmap.png",
        show=False,
    )

    print("  Saved: explainability_drift_heatmap.png")
    print("  Saved: explainability_nodrift_heatmap.png")
    plt.show()
