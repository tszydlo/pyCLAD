import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from pyclad.callbacks.callback import Callback


class DriftMetricsCallback(Callback):
    """Computes cosine similarity and soft top-k churn between consecutive concept importances."""

    def __init__(self, mc_callback, concept_signals, k=5):
        """Initialise the drift metrics callback.

        Args:
            mc_callback: A ModelComparisonCallback whose rf_importances are read each step.
            concept_signals: Mapping of concept name to ground-truth signal feature indices.
            k: Number of top features used for churn calculation.
        """
        self._mc = mc_callback
        self._signals = concept_signals
        self._k = k
        self._history = []
        self.results = {}

    def after_training(self, learned_concept):
        """Compute cosine similarity and soft churn between this concept and the previous one.

        Results are stored in self.results keyed by concept name.
        Skips the first concept as there is no previous state to compare against.

        Args:
            learned_concept: The concept that was just trained by the strategy.
        """
        name = learned_concept.name
        if name not in self._mc.rf_importances:
            return

        fi = self._mc.rf_importances[name]
        self._history.append((name, fi))
        if len(self._history) < 2:
            return

        prev_name, fi_prev = self._history[-2]

        cos_sim = float(cosine_similarity([fi_prev], [fi])[0, 0])

        prev_top = set(np.argsort(fi_prev)[-self._k :])
        curr_top = set(np.argsort(fi)[-self._k :])
        hard_churn = len(prev_top.symmetric_difference(curr_top)) / (2 * self._k)
        importance_shift = float(np.sum(np.abs(fi_prev - fi)))
        soft_churn = hard_churn * importance_shift

        # Ground truth label for verification only, not used in detection
        overlap = len(set(self._signals.get(prev_name, [])) & set(self._signals.get(name, [])))
        expected = "HIGH" if overlap == 0 else ("MODERATE" if overlap == 1 else "LOW")

        self.results[name] = {
            "cos_sim": cos_sim,
            "churn": soft_churn,
            "expected_drift": expected,
        }
        print(f"    [drift {prev_name}→{name}] cos={cos_sim:.3f}  churn={soft_churn:.3f}  gt={expected}")
