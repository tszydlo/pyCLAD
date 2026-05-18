import numpy as np
from scipy.stats import wasserstein_distance

from pyclad.callbacks.callback import Callback


class WassersteinDriftCallback(Callback):
    """Measures per-feature Wasserstein distance between consecutive buffer snapshots."""

    def __init__(self, strategy, feature_names):
        """Initialise the Wasserstein drift callback.

        Args:
            strategy: A ReplayEnhancedStrategy whose buffer is read after each concept.
            feature_names: List of feature name strings for the results dict keys.
        """
        self._strategy = strategy
        self._feature_names = feature_names
        self._prev_snapshot = None
        self.results = {}

    def after_training(self, learned_concept):
        """Compute per-feature Earth Mover's Distance between the current and previous buffer.

        Results are stored in self.results keyed by concept name, containing
        mean_wasserstein, max_wasserstein, and per_feature distances.
        Skips the first concept as there is no previous snapshot to compare against.

        Args:
            learned_concept: The concept that was just trained by the strategy.
        """
        buf = self._strategy._buffer.data()

        if self._prev_snapshot is not None and len(buf) > 0:
            w_distances = [wasserstein_distance(self._prev_snapshot[:, f], buf[:, f]) for f in range(buf.shape[1])]
            self.results[learned_concept.name] = {
                "mean_wasserstein": float(np.mean(w_distances)),
                "max_wasserstein": float(np.max(w_distances)),
                "per_feature": dict(zip(self._feature_names, w_distances)),
            }

        # Always update snapshot — must stay outside the if block
        self._prev_snapshot = buf.copy() if len(buf) > 0 else None
