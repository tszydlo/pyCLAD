import numpy as np

from pyclad.models.adapters.adaptive_threshold import AdaptiveThresholdDetector
from pyclad.models.model import Model


def test_adaptive_threshold_detector_contract():
    model = AdaptiveThresholdDetector()
    assert isinstance(model, Model)

    X_train_norm = np.random.randn(50, 8)
    model.fit(X_train_norm)
    assert model.threshold_ != 0.0

    X_calib = np.vstack([np.random.randn(20, 8), np.random.randn(5, 8) + 3.0])
    y_calib = np.array([0] * 20 + [1] * 5)
    model.calibrate(X_calib, y_calib)
    assert model.calibrated_ is True

    X_test = np.random.randn(15, 8)
    preds, scores = model.predict(X_test)
    assert preds.shape == (15,)
    assert scores.shape == (15,)
    assert set(np.unique(preds)).issubset({0, 1})
