import numpy as np
import pandas as pd

from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset


class SyntheticConceptGenerator:
    """Generates synthetic concepts where anomalies have shifted signal features."""

    def __init__(
        self,
        concept_signals: dict,
        n_features: int = 20,
        n_train: int = 150,
        n_test_normal: int = 60,
        n_test_anomaly: int = 40,
        signal_strength: float = 5.0,
        seed: int = 42,
    ):
        """Initialise the generator.

        Args:
            concept_signals: Mapping of concept name to list of signal feature indices.
            n_features: Total number of features per sample.
            n_train: Number of training samples per concept.
            n_test_normal: Number of normal test samples per concept.
            n_test_anomaly: Number of anomalous test samples per concept.
            signal_strength: Mean shift applied to anomaly signal features.
            seed: Base random seed; each concept uses seed + offset.
        """
        self._concept_signals = concept_signals
        self._n_features = n_features
        self._n_train = n_train
        self._n_test_normal = n_test_normal
        self._n_test_anomaly = n_test_anomaly
        self._signal_strength = signal_strength
        self._seed = seed
        self._train_concepts = []
        self._test_concepts = []
        self._all_data_records = []
        self._dataset_built = False

    def _generate_concept_data(self, active_features, seed_offset):
        """Generate train and test arrays for one concept.

        Args:
            active_features: Feature indices that are shifted for anomalies.
            seed_offset: Added to base seed to make each concept reproducible independently.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test) as numpy arrays.
        """
        rng = np.random.RandomState(self._seed + seed_offset)

        # 5% contamination for training
        n_anomaly = int(self._n_train * 0.05)
        n_normal = self._n_train - n_anomaly

        X_n = rng.randn(n_normal, self._n_features)
        X_a = rng.randn(n_anomaly, self._n_features)
        for f in active_features:
            X_a[:, f] += self._signal_strength

        X_train = np.vstack([X_n, X_a])
        y_train = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
        idx = rng.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]

        # Test set is not shuffled
        X_tn = rng.randn(self._n_test_normal, self._n_features)
        X_ta = rng.randn(self._n_test_anomaly, self._n_features)
        for f in active_features:
            X_ta[:, f] += self._signal_strength

        X_test = np.vstack([X_tn, X_ta])
        y_test = np.array([0] * self._n_test_normal + [1] * self._n_test_anomaly)

        return X_train, y_train, X_test, y_test

    def build_dataset(self) -> ConceptsDataset:
        """Build and return a ConceptsDataset from all configured concepts.

        Returns:
            A ConceptsDataset containing one Concept per entry in concept_signals.
        """
        self._train_concepts = []
        self._test_concepts = []
        self._all_data_records = []

        for idx, (name, active_features) in enumerate(self._concept_signals.items()):
            X_tr, y_tr, X_te, y_te = self._generate_concept_data(active_features, seed_offset=idx * 10)

            self._train_concepts.append(Concept(name, data=X_tr, labels=y_tr))
            self._test_concepts.append(Concept(name, data=X_te, labels=y_te))

            for i in range(len(X_tr)):
                row = {f"f{k}": X_tr[i, k] for k in range(self._n_features)}
                row.update({"concept": name, "split": "train", "label": int(y_tr[i])})
                self._all_data_records.append(row)

            for i in range(len(X_te)):
                row = {f"f{k}": X_te[i, k] for k in range(self._n_features)}
                row.update({"concept": name, "split": "test", "label": int(y_te[i])})
                self._all_data_records.append(row)

            print(f"  {name}: train={X_tr.shape}  active={[f'f{f}' for f in active_features]}")

        self._dataset_built = True
        dataset = ConceptsDataset(
            name="SyntheticDataset",
            train_concepts=self._train_concepts,
            test_concepts=self._test_concepts,
        )
        print(f"\nDataset built: {len(self._train_concepts)} concepts, {self._n_features} features")
        return dataset

    def export_csv(self, path: str) -> None:
        """Export the full dataset (train + test, all concepts) to a single CSV.

        Args:
            path: File path to write the CSV to.
        """
        self._check_built()
        pd.DataFrame(self._all_data_records).to_csv(path, index=False)
        print(f"Saved: {path}")

    def export_splits(self, train_path: str, test_path: str) -> None:
        """Export train and test splits to separate CSV files.

        Args:
            train_path: File path for the training split CSV.
            test_path: File path for the test split CSV.
        """
        self._check_built()
        df = pd.DataFrame(self._all_data_records)
        df[df["split"] == "train"].drop(columns=["split"]).to_csv(train_path, index=False)
        df[df["split"] == "test"].drop(columns=["split"]).to_csv(test_path, index=False)
        print(f"Saved: {train_path}  |  {test_path}")

    @property
    def feature_names(self):
        """List of feature names in order (f0, f1, ..., fN-1)."""
        return [f"f{i}" for i in range(self._n_features)]

    @property
    def concept_order(self):
        """Ordered list of concept names as defined in concept_signals."""
        return list(self._concept_signals.keys())

    @property
    def concept_signals(self):
        """The concept_signals dict passed at construction."""
        return self._concept_signals

    def _check_built(self):
        """Raise if build_dataset() has not been called yet."""
        if not self._dataset_built:
            raise RuntimeError("Call build_dataset() first.")
