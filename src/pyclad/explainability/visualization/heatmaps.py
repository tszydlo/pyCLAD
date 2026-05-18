import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureImportanceHeatmap:
    """Heatmap of per-concept RF feature importances for all features f0–fN."""

    def __init__(self, concept_order, concept_signals, feature_names, n_features):
        """Initialise the heatmap visualiser.

        Args:
            concept_order: Ordered list of concept names (x-axis).
            concept_signals: Mapping of concept name to ground-truth signal feature indices.
            feature_names: List of feature name strings, e.g. ['f0', 'f1', ...].
            n_features: Total number of features to display.
        """
        self._concept_order = concept_order
        self._concept_signals = concept_signals
        self._feature_names = feature_names
        self._n_features = n_features

    def plot(self, importances_dict, scenario_name, n_top=5, save_path=None, show=True, prefix=""):
        """Render the feature importance heatmap.

        Args:
            importances_dict: Dict mapping concept name to a numpy array of feature importances.
            scenario_name: Label shown in the plot title.
            n_top: Kept for API compatibility; all features are shown with their true values.
            save_path: Optional file path to save the figure (PNG/PDF).
            show: If True, call plt.show() to display the figure interactively.
            prefix: Optional filename prefix (unused in current implementation).
        """
        n = len(self._concept_order)

        # Full importance matrix — actual RF values for every feature, every concept
        imp = np.zeros((n, self._n_features))
        for i, c in enumerate(self._concept_order):
            if c in importances_dict:
                imp[i] = importances_dict[c]

        # Label all features; append X to ground-truth signal features
        all_signals = {f for feats in self._concept_signals.values() for f in feats}
        all_names = [
            f"{self._feature_names[k]} X" if k in all_signals else self._feature_names[k]
            for k in range(self._n_features)
        ]

        # Transpose so features are rows (f0 at top) and concepts are columns
        data = imp.T
        annot = np.round(data, 3).astype(str).tolist()

        fig, ax = plt.subplots(figsize=(13, max(8, self._n_features * 0.45 + 2)))
        sns.heatmap(
            data,
            xticklabels=self._concept_order,
            yticklabels=all_names,
            cmap="YlOrRd",
            ax=ax,
            linewidths=0.4,
            annot=annot,
            fmt="",
            annot_kws={"size": 9},
            vmin=0,
            cbar_kws={"label": "RF Importance"},
        )
        ax.set_title(
            f"RF Feature Importance — All Features\n"
            f"Scenario: {scenario_name}  |  X = ground-truth signal feature  |  seed=42",
            fontsize=12,
        )
        ax.set_xlabel("Concept")
        ax.set_ylabel("Feature")
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        if show:
            plt.show()
        plt.close(fig)
