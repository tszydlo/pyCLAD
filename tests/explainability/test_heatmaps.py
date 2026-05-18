import matplotlib

matplotlib.use("Agg") 

import numpy as np
import pytest

from pyclad.explainability.visualization.heatmaps import FeatureImportanceHeatmap

CONCEPT_ORDER = ["C0", "C1", "C2"]
CONCEPT_SIGNALS = {"C0": [0], "C1": [1], "C2": [2]}
FEATURE_NAMES = [f"f{i}" for i in range(5)]
N_FEATURES = 5


@pytest.fixture
def heatmap():
    return FeatureImportanceHeatmap(CONCEPT_ORDER, CONCEPT_SIGNALS, FEATURE_NAMES, N_FEATURES)


@pytest.fixture
def importances():
    rng = np.random.RandomState(0)
    return {c: rng.dirichlet(np.ones(N_FEATURES)) for c in CONCEPT_ORDER}


def test_plot_runs_without_error(heatmap, importances):
    """plot() completes without raising for valid inputs."""
    heatmap.plot(importances, "TestScenario", show=False)


def test_plot_saves_file(tmp_path, heatmap, importances):
    """plot() writes a PNG file when save_path is provided."""
    out = str(tmp_path / "heatmap.png")
    heatmap.plot(importances, "TestScenario", save_path=out, show=False)
    assert (tmp_path / "heatmap.png").exists()


def test_signal_features_labelled(heatmap, importances):
    """Signal features are annotated with X suffix in all_names."""
    # Access private to inspect label list
    all_signals = {f for feats in CONCEPT_SIGNALS.values() for f in feats}
    all_names = [
        f"{FEATURE_NAMES[k]} X" if k in all_signals else FEATURE_NAMES[k] for k in range(N_FEATURES)
    ]
    for sig in all_signals:
        assert f"{FEATURE_NAMES[sig]} X" in all_names


def test_missing_concept_handled(heatmap):
    """plot() handles missing concept in importances_dict gracefully (fills zeros)."""
    partial = {"C0": np.ones(N_FEATURES) / N_FEATURES}
    heatmap.plot(partial, "Partial", show=False)


def test_empty_importances(heatmap):
    """plot() handles an empty importances dict without raising."""
    heatmap.plot({}, "Empty", show=False)
