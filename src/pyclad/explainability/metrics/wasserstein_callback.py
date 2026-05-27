import numpy as np
from scipy.stats import wasserstein_distance

from pyclad.callbacks.callback import Callback


class WassersteinDriftCallback(Callback):
    """Measures per-feature Wasserstein distance between consecutive buffer snapshots.

    Args:
        strategy: A ``ReplayEnhancedStrategy`` whose buffer is read after each concept.
        feature_names: List of feature-name strings used as keys in the per-feature
            results dict.
        mc_callback: Optional ``ModelComparisonCallback`` (or any object that exposes
            a ``rf_importances`` dict keyed by concept name).  When supplied, the
            dot product of those importances with the raw per-feature EMD values is
            stored under the ``'weighted_wasserstein'`` key.  Defaults to ``None``
            (raw-only mode).

    Attributes:
        results (dict): Keyed by concept name.  Each entry always contains:

            - ``mean_wasserstein`` (*float*): Mean EMD across all features.
            - ``max_wasserstein``  (*float*): Max  EMD across all features.
            - ``per_feature``      (*dict*):  ``{feature_name: raw_emd}`` mapping.

    """

    def __init__(self, strategy, feature_names, mc_callback=None):
        """Initialise the Wasserstein drift callback.

        Args:
            strategy: A ReplayEnhancedStrategy whose buffer is read after each concept.
            feature_names: List of feature name strings for the results dict keys.
            mc_callback: Optional ModelComparisonCallback (or compatible object).
                When supplied, also computes importance-weighted Wasserstein.
                Defaults to None (raw-only mode, backward-compatible).
        """
        self._strategy      = strategy
        self._feature_names = feature_names
        self._mc            = mc_callback  
        self._prev_snapshot = None
        self.results        = {}

    def after_training(self, learned_concept):
        """Compute per-feature EMD between the current and previous buffer snapshots.

        Args:
            learned_concept: The concept that was just trained by the strategy.
        """
        buf = self._strategy._buffer.data()

        if self._prev_snapshot is not None and len(buf) > 0:

            #  Raw per-feature EMD
            raw_emd = [
                wasserstein_distance(self._prev_snapshot[:, f], buf[:, f])
                for f in range(buf.shape[1])
            ]

            entry = {
                "mean_wasserstein": float(np.mean(raw_emd)),
                "max_wasserstein":  float(np.max(raw_emd)),
                "per_feature":      dict(zip(self._feature_names, raw_emd)),
            }

            # Importance-weighted EMD
            if self._mc is not None:
                importances = self._mc.rf_importances.get(learned_concept.name)
                if importances is not None:
                    entry["weighted_wasserstein"] = float(np.dot(importances, raw_emd))

            self.results[learned_concept.name] = entry

        self._prev_snapshot = buf.copy() if len(buf) > 0 else None
