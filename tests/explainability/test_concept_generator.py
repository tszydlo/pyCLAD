import os
import tempfile

import numpy as np
import pytest

from pyclad.explainability.data_generation.concept_generator import SyntheticConceptGenerator


SIGNALS = {"C0": [0, 1], "C1": [2, 3], "C2": [4, 5]}
SMALL = SyntheticConceptGenerator(concept_signals=SIGNALS, n_features=10, n_train=40, n_test_normal=10, n_test_anomaly=5)


@pytest.fixture
def dataset():
    return SMALL.build_dataset()


def test_build_dataset_concept_count(dataset):
    """build_dataset returns the correct number of concepts."""
    assert len(dataset.train_concepts()) == 3
    assert len(dataset.test_concepts()) == 3


def test_build_dataset_train_shape(dataset):
    """Each training concept has the correct number of rows and features."""
    for c in dataset.train_concepts():
        assert c.data.shape == (40, 10)
        assert len(c.labels) == 40


def test_build_dataset_test_shape(dataset):
    """Each test concept has the correct number of rows and features."""
    for c in dataset.test_concepts():
        assert c.data.shape == (15, 10)
        assert len(c.labels) == 15


def test_train_contamination():
    """Training set contains ~5% anomaly labels."""
    ds = SMALL.build_dataset()
    for c in ds.train_concepts():
        ratio = c.labels.mean()
        assert 0.0 < ratio <= 0.10, f"Unexpected contamination: {ratio}"


def test_test_labels_unsorted():
    """Test labels are [0...0, 1...1] — not shuffled."""
    ds = SMALL.build_dataset()
    for c in ds.test_concepts():
        n_normal = (c.labels == 0).sum()
        assert list(c.labels[:n_normal]) == [0] * n_normal
        assert list(c.labels[n_normal:]) == [1] * (len(c.labels) - n_normal)


def test_feature_names():
    """feature_names returns f0..fN-1."""
    gen = SyntheticConceptGenerator({"A": [0]}, n_features=5)
    assert gen.feature_names == ["f0", "f1", "f2", "f3", "f4"]


def test_concept_order():
    """concept_order preserves insertion order."""
    gen = SyntheticConceptGenerator({"X": [0], "Y": [1], "Z": [2]}, n_features=5)
    assert gen.concept_order == ["X", "Y", "Z"]


def test_export_csv(tmp_path, dataset):
    """export_csv creates a file with the expected number of rows."""
    path = str(tmp_path / "out.csv")
    SMALL.export_csv(path)
    import pandas as pd

    df = pd.read_csv(path)
    expected_rows = 3 * (40 + 15)
    assert len(df) == expected_rows
    assert "concept" in df.columns
    assert "label" in df.columns


def test_export_splits(tmp_path, dataset):
    """export_splits creates separate train/test CSVs without a split column."""
    tr = str(tmp_path / "train.csv")
    te = str(tmp_path / "test.csv")
    SMALL.export_splits(tr, te)
    import pandas as pd

    df_tr = pd.read_csv(tr)
    df_te = pd.read_csv(te)
    assert len(df_tr) == 3 * 40
    assert len(df_te) == 3 * 15
    assert "split" not in df_tr.columns
    assert "split" not in df_te.columns


def test_export_requires_build():
    """export_csv raises RuntimeError if called before build_dataset."""
    gen = SyntheticConceptGenerator({"A": [0]}, n_features=5)
    with pytest.raises(RuntimeError):
        gen.export_csv("irrelevant.csv")


def test_reproducibility():
    """Same seed produces identical data across two calls."""
    g1 = SyntheticConceptGenerator({"A": [0]}, n_features=5, seed=99)
    g2 = SyntheticConceptGenerator({"A": [0]}, n_features=5, seed=99)
    ds1 = g1.build_dataset()
    ds2 = g2.build_dataset()
    np.testing.assert_array_equal(ds1.train_concepts()[0].data, ds2.train_concepts()[0].data)
