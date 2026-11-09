from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import balanced_accuracy_score

from pyclad.models.model import Model


class AdaptiveThresholdDetector(Model):
    """
    Adaptive Threshold Anomaly Detector
    """

    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        percentiles: Optional[List[int]] = None,
        random_state: int = 42,
    ) -> None:
        self.estimator = base_estimator or IsolationForest(n_estimators=100, random_state=random_state, n_jobs=-1)
        self.percentiles = percentiles or list(range(50, 95, 5))
        self.threshold_: float = 0.0
        self.calibrated_: bool = False

    def fit(self, data: np.ndarray) -> None:
        self.estimator.fit(data)
        scores = self.decision_function(data)
        self.threshold_ = float(np.percentile(scores, 90))

    def decision_function(self, data: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "score_samples"):
            return -self.estimator.score_samples(data)
        return self.estimator.decision_function(data)

    def calibrate(self, x_calib: np.ndarray, y_calib: np.ndarray) -> float:
        """
        Dynamically optimize decision threshold on labeled calibration telemetry.
        """
        scores = self.decision_function(x_calib)
        if len(np.unique(y_calib)) >= 2:
            best_score = -1.0
            best_th = self.threshold_
            for p in self.percentiles:
                th_c = float(np.percentile(scores, p))
                preds = (scores >= th_c).astype(int)
                score = balanced_accuracy_score(y_calib, preds)
                if score > best_score:
                    best_score = score
                    best_th = th_c
            self.threshold_ = best_th
        else:
            self.threshold_ = float(np.percentile(scores, 90))

        self.calibrated_ = True
        return self.threshold_

    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict binary labels and anomaly scores matching pyCLAD Model contract.
        """
        scores = self.decision_function(data)
        preds = (scores >= self.threshold_).astype(int)
        return preds, scores

    def name(self) -> str:
        return "AdaptiveThresholdDetector"

    def additional_info(self) -> Dict[str, Any]:
        return {"threshold": self.threshold_, "calibrated": self.calibrated_}
