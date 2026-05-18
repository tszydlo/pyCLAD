import numpy as np
import pytest

from pyclad.explainability.metrics.drift_metrics_callback import DriftMetricsCallback
from pyclad.explainability.metrics.wasserstein_callback import WassersteinDriftCallback


# Shared helpers


CONCEPT_SIGNALS = {"C0": [0, 1], "C1": [2, 3], "C2": [0, 1]}
FEATURE_NAMES = [f"f{i}" for i in range(5)]
N_FEATURES = 5


class _FakeImportances:
    """Minimal stand-in for ModelComparisonCallback with pre-set importances."""

    def __init__(self, importances: dict):
        self.rf_importances = importances


class _FakeConcept:
    def __init__(self, name):
        self.name = name


class _FakeBuffer:
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


class _FakeStrategy:
    def __init__(self, data):
        self._buffer = _FakeBuffer(data)



# DriftMetricsCallback

def test_drift_metrics_first_concept_skipped():
    """No results are stored for the first concept (nothing to compare against)."""
    fi = {"C0": np.array([0.5, 0.5, 0.0, 0.0, 0.0])}
    mc = _FakeImportances(fi)
    cb = DriftMetricsCallback(mc, CONCEPT_SIGNALS, k=2)
    cb.after_training(_FakeConcept("C0"))
    assert len(cb.results) == 0


def test_drift_metrics_second_concept_populated():
    """Results are populated for the second concept onwards."""
    fi = {
        "C0": np.array([0.5, 0.5, 0.0, 0.0, 0.0]),
        "C1": np.array([0.0, 0.0, 0.5, 0.5, 0.0]),
    }
    mc = _FakeImportances(fi)
    cb = DriftMetricsCallback(mc, CONCEPT_SIGNALS, k=2)
    cb.after_training(_FakeConcept("C0"))
    cb.after_training(_FakeConcept("C1"))
    assert "C1" in cb.results
    assert "cos_sim" in cb.results["C1"]
    assert "churn" in cb.results["C1"]
    assert "expected_drift" in cb.results["C1"]


def test_drift_metrics_identical_importances_cos_sim_is_one():
    """Two identical importance vectors yield cosine similarity of 1.0."""
    imp = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    fi = {"C0": imp.copy(), "C1": imp.copy()}
    mc = _FakeImportances(fi)
    cb = DriftMetricsCallback(mc, CONCEPT_SIGNALS, k=2)
    cb.after_training(_FakeConcept("C0"))
    cb.after_training(_FakeConcept("C1"))
    assert pytest.approx(cb.results["C1"]["cos_sim"], abs=1e-6) == 1.0


def test_drift_metrics_identical_importances_churn_is_zero():
    """Identical importance vectors yield zero churn."""
    imp = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    fi = {"C0": imp.copy(), "C1": imp.copy()}
    mc = _FakeImportances(fi)
    cb = DriftMetricsCallback(mc, CONCEPT_SIGNALS, k=2)
    cb.after_training(_FakeConcept("C0"))
    cb.after_training(_FakeConcept("C1"))
    assert cb.results["C1"]["churn"] == pytest.approx(0.0, abs=1e-6)


def test_drift_metrics_expected_drift_high():
    """Concepts with no shared signals are labelled HIGH."""
    signals = {"C0": [0], "C1": [4]}
    fi = {"C0": np.eye(5)[0], "C1": np.eye(5)[4]}
    mc = _FakeImportances(fi)
    cb = DriftMetricsCallback(mc, signals, k=1)
    cb.after_training(_FakeConcept("C0"))
    cb.after_training(_FakeConcept("C1"))
    assert cb.results["C1"]["expected_drift"] == "HIGH"


def test_drift_metrics_expected_drift_low():
    """Concepts with fully shared signals are labelled LOW."""
    signals = {"C0": [0, 1], "C1": [0, 1]}
    fi = {"C0": np.array([0.5, 0.5, 0.0, 0.0, 0.0]), "C1": np.array([0.5, 0.5, 0.0, 0.0, 0.0])}
    mc = _FakeImportances(fi)
    cb = DriftMetricsCallback(mc, signals, k=2)
    cb.after_training(_FakeConcept("C0"))
    cb.after_training(_FakeConcept("C1"))
    assert cb.results["C1"]["expected_drift"] == "LOW"


def test_drift_metrics_skips_unknown_concept():
    """Concepts missing from rf_importances are silently skipped."""
    mc = _FakeImportances({})
    cb = DriftMetricsCallback(mc, CONCEPT_SIGNALS, k=2)
    cb.after_training(_FakeConcept("C0"))
    assert len(cb.results) == 0


# WassersteinDriftCallback



def test_wasserstein_first_concept_no_results():
    """No results are stored for the first concept."""
    data = np.random.randn(50, N_FEATURES)
    strategy = _FakeStrategy(data)
    cb = WassersteinDriftCallback(strategy, FEATURE_NAMES)
    cb.after_training(_FakeConcept("C0"))
    assert len(cb.results) == 0


def test_wasserstein_second_concept_populated():
    """Results are stored from the second concept onwards."""
    rng = np.random.RandomState(0)
    strategy = _FakeStrategy(rng.randn(50, N_FEATURES))
    cb = WassersteinDriftCallback(strategy, FEATURE_NAMES)
    cb.after_training(_FakeConcept("C0"))
    strategy._buffer = _FakeBuffer(rng.randn(50, N_FEATURES))
    cb.after_training(_FakeConcept("C1"))
    assert "C1" in cb.results
    assert cb.results["C1"]["mean_wasserstein"] >= 0.0
    assert cb.results["C1"]["max_wasserstein"] >= cb.results["C1"]["mean_wasserstein"]


def test_wasserstein_identical_buffers_is_zero():
    """Identical consecutive buffers yield Wasserstein distances of zero."""
    data = np.random.RandomState(7).randn(50, N_FEATURES)
    strategy = _FakeStrategy(data.copy())
    cb = WassersteinDriftCallback(strategy, FEATURE_NAMES)
    cb.after_training(_FakeConcept("C0"))
    strategy._buffer = _FakeBuffer(data.copy())
    cb.after_training(_FakeConcept("C1"))
    assert cb.results["C1"]["mean_wasserstein"] == pytest.approx(0.0, abs=1e-6)


def test_wasserstein_shifted_buffer_has_positive_distance():
    """Clearly shifted buffers yield a positive Wasserstein distance."""
    rng = np.random.RandomState(1)
    buf_a = rng.randn(50, N_FEATURES)
    buf_b = rng.randn(50, N_FEATURES) + 10.0
    strategy = _FakeStrategy(buf_a)
    cb = WassersteinDriftCallback(strategy, FEATURE_NAMES)
    cb.after_training(_FakeConcept("C0"))
    strategy._buffer = _FakeBuffer(buf_b)
    cb.after_training(_FakeConcept("C1"))
    assert cb.results["C1"]["mean_wasserstein"] > 1.0


def test_wasserstein_per_feature_keys(dataset=None):
    """per_feature dict contains one key per feature name."""
    rng = np.random.RandomState(2)
    strategy = _FakeStrategy(rng.randn(30, N_FEATURES))
    cb = WassersteinDriftCallback(strategy, FEATURE_NAMES)
    cb.after_training(_FakeConcept("C0"))
    strategy._buffer = _FakeBuffer(rng.randn(30, N_FEATURES))
    cb.after_training(_FakeConcept("C1"))
    assert set(cb.results["C1"]["per_feature"].keys()) == set(FEATURE_NAMES)
