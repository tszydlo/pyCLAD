import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from pyclad.callbacks.callback import Callback
from pyclad.data.datasets.opssat_dataset import OpsSatDataset
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.baselines.cumulative import CumulativeStrategy

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CHANNELS = [
    "CADC0872",
    "CADC0873",
    "CADC0874",
    "CADC0884",
    "CADC0886",
    "CADC0888",
    "CADC0890",
    "CADC0892",
    "CADC0894",
]


class TrackerCallback(Callback):
    """Callback to record step-by-step ROC-AUC, balanced accuracy, and F1 scores."""

    def __init__(self):
        self.auc_history = []
        self.balanced_accuracy_history = []
        self.f1_history = []

    def after_evaluation(self, evaluated_concept, y_true, y_pred, anomaly_scores, *args, **kwargs):
        self.auc_history.append(roc_auc_score(y_true, anomaly_scores))
        self.balanced_accuracy_history.append(balanced_accuracy_score(y_true, y_pred))
        self.f1_history.append(f1_score(y_true, y_pred))


def plot_metric_histories(metric_histories):
    channels = list(metric_histories.keys())
    ncols = 3
    nrows = math.ceil(len(channels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for ax, ch in zip(axes.flat, channels):
        for metric_name, history in metric_histories[ch].items():
            ax.plot(history, label=metric_name)
        ax.set_title(ch)
        ax.set_xlabel("Evaluation step")
        ax.set_ylabel("Score")
        ax.legend()

    for ax in axes.flat[len(channels):]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()


def run_auc_benchmark():
    print("=" * 68)
    print(f"{'Channel':<12} | {'Concepts':<10} | {'CA_AUC':<10} | {'Final_AUC':<10} | {'Time (s)':<8}")
    print("=" * 68)

    all_ca_auc = []
    all_final_auc = []
    metric_histories = {}

    for ch in CHANNELS:
        t0 = time.time()
        dataset = OpsSatDataset(channel=ch, includes_anomaly=False)
        print(f"Dataset info: {dataset.info()}")
        model = IsolationForestAdapter(contamination=0.1, n_estimators=100, random_state=SEED)
        strategy = CumulativeStrategy(model=model)
        tracker = TrackerCallback()

        scenario = ConceptIncrementalScenario(
            dataset=dataset,
            strategy=strategy,
            callbacks=[tracker],
        )
        scenario.run()
        metric_histories[ch] = {
            "ROC-AUC": tracker.auc_history,
            "Balanced Accuracy": tracker.balanced_accuracy_history,
            "F1": tracker.f1_history,
        }

        elapsed = round(time.time() - t0, 2)
        ca_auc = round(float(np.mean(tracker.auc_history)), 4)
        final_auc = round(float(tracker.auc_history[-1]), 4)

        all_ca_auc.append(ca_auc)
        all_final_auc.append(final_auc)

        print(f"{ch:<12} | {len(dataset.train_concepts()):<10} | {ca_auc:<10.4f} | {final_auc:<10.4f} | {elapsed:<8.2f}")

    print("=" * 68)
    print(f"Mean CA_AUC across channels: {np.mean(all_ca_auc):.4f}")
    print(f"Mean Final_AUC across channels: {np.mean(all_final_auc):.4f}")

    plot_metric_histories(metric_histories)


if __name__ == "__main__":
    run_auc_benchmark()
