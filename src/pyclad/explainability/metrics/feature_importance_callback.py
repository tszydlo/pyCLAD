import numpy as np
from sklearn.ensemble import RandomForestClassifier

from pyclad.callbacks.callback import Callback
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter


class ModelComparisonCallback(Callback):
    """Scores the replay buffer with IF, then uses RF to extract feature importances."""

    def __init__(self, strategy, feature_names, seed=42):
        """Initialise the callback.

        Args:
            strategy: A ReplayEnhancedStrategy whose buffer is read after each concept.
            feature_names: List of feature name strings for reporting.
            seed: Random seed for both IF and RF reproducibility.
        """
        self._strategy = strategy
        self._feature_names = feature_names
        self._seed = seed
        self._seen_concepts = []
        self.rf_importances = {}

    def after_training(self, learned_concept):
        """Run the IF+RF surrogate pipeline on the current buffer contents.

        Isolation Forest scores the buffer; the top 10th-percentile become
        pseudo-anomaly labels used to supervise a Random Forest, whose
        feature importances are stored in rf_importances.

        Args:
            learned_concept: The concept that was just trained by the strategy.
        """
        self._seen_concepts.append(learned_concept.name)
        buf = self._strategy._buffer.data()
        if len(buf) == 0:
            return

        iso = IsolationForestAdapter(contamination=0.1, random_state=self._seed)
        iso.fit(buf)
        _, scores = iso.predict(buf)

        # Top 10% become pseudo-anomalies (label=1), rest are pseudo-normal
        pseudo = (scores >= np.percentile(scores, 90)).astype(int)
        if len(np.unique(pseudo)) > 1:
            rf = RandomForestClassifier(n_estimators=200, random_state=self._seed, n_jobs=-1)
            rf.fit(buf, pseudo)
            self.rf_importances[learned_concept.name] = rf.feature_importances_
            top2 = [self._feature_names[i] for i in np.argsort(rf.feature_importances_)[-2:]]
            print(f"    [{learned_concept.name}] Top-2: {sorted(top2)}")
