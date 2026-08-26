import random
import time

import numpy as np
from sklearn.metrics import roc_auc_score

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


class AucTrackerCallback(Callback):
    """Callback to record step-by-step test ROC-AUC scores."""

    def __init__(self):
        self.auc_history = []

    def after_evaluation(self, evaluated_concept, y_true, y_pred, anomaly_scores, *args, **kwargs):
        auc = roc_auc_score(y_true, anomaly_scores)
        self.auc_history.append(auc)


def run_auc_benchmark():
    print("=" * 68)
    print(f"{'Channel':<12} | {'Concepts':<10} | {'CA_AUC':<10} | {'Final_AUC':<10} | {'Time (s)':<8}")
    print("=" * 68)

    all_ca_auc = []
    all_final_auc = []

    for ch in CHANNELS:
        t0 = time.time()
        dataset = OpsSatDataset(channel=ch, includes_anomaly=True)
        print(f"Dataset info: {dataset.info()}")
        model = IsolationForestAdapter(contamination=0.1, n_estimators=100, random_state=SEED)
        strategy = CumulativeStrategy(model=model)
        tracker = AucTrackerCallback()

        scenario = ConceptIncrementalScenario(
            dataset=dataset,
            strategy=strategy,
            callbacks=[tracker],
        )
        scenario.run()

        elapsed = round(time.time() - t0, 2)
        ca_auc = round(float(np.mean(tracker.auc_history)), 4)
        final_auc = round(float(tracker.auc_history[-1]), 4)

        all_ca_auc.append(ca_auc)
        all_final_auc.append(final_auc)

        print(f"{ch:<12} | {len(dataset.train_concepts()):<10} | {ca_auc:<10.4f} | {final_auc:<10.4f} | {elapsed:<8.2f}")

    print("=" * 68)
    print(f"Mean CA_AUC across channels: {np.mean(all_ca_auc):.4f}")
    print(f"Mean Final_AUC across channels: {np.mean(all_final_auc):.4f}")


if __name__ == "__main__":
    run_auc_benchmark()
