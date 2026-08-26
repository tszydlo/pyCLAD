from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.readers.concepts_readers import read_concepts_from_df

OPS_SAT_CHANNELS_TYPE = Literal[
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


class OpsSatDataset(ConceptsDataset):
    """
    The OPS-SAT satellite telemetry dataset contains continuous operational telemetry
    from the European Space Agency (ESA) OPS-SAT CubeSat mission.
    """

    def __init__(
        self,
        channel: OPS_SAT_CHANNELS_TYPE = "CADC0874",
        includes_anomaly: bool = False,
        data_dir: Optional[Union[str, Path]] = None,
    ):
        """
        :param channel: The telemetry channel identifier.
        :param includes_anomaly: If True, keep anomaly-labeled rows in the training set. If False (default),
            exclude anomaly-labeled rows so training data contains only normal samples.
        :param data_dir: Directory containing the processed train/test CSV files.
        """
        if data_dir is None:
            candidates = [
                Path("dataset/opssat_lad_data"),
                Path("pyCLAD-main/dataset/opssat_lad_data"),
                Path(__file__).resolve().parents[4] / "dataset" / "opssat_lad_data",
                Path("Datasets/ops_sat_pyclad"),
                Path("../Datasets/ops_sat_pyclad"),
            ]
            data_dir = next((p for p in candidates if (p / f"{channel}_train.csv").exists()), candidates[0])
        else:
            data_dir = Path(data_dir)

        train_file = data_dir / f"{channel}_train.csv"
        test_file = data_dir / f"{channel}_test.csv"

        if not train_file.exists() or not test_file.exists():
            raise FileNotFoundError(
                f"OPS-SAT files for channel '{channel}' not found in '{data_dir}'. "
                f"Ensure '{channel}_train.csv' and '{channel}_test.csv' exist."
            )

        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        if "concept_id" not in train_df.columns:
            train_df.insert(0, "concept_id", np.arange(len(train_df)))
        if "label" not in train_df.columns:
            train_df.insert(2, "label", 0)

        if not includes_anomaly:
            train_df = train_df[train_df["label"] == 0].reset_index(drop=True)
            train_df["concept_id"] = np.arange(len(train_df))

        if "concept_id" not in test_df.columns:
            test_df.insert(0, "concept_id", 0)
        if "label" not in test_df.columns:
            test_df.insert(2, "label", test_df["anomaly"] if "anomaly" in test_df.columns else 0)

        train_concepts = read_concepts_from_df(train_df)
        test_concepts = read_concepts_from_df(test_df)

        super().__init__(
            name=f"OPS-SAT-{channel}",
            train_concepts=train_concepts,
            test_concepts=test_concepts,
        )


if __name__ == "__main__":
    dataset = OpsSatDataset(channel="CADC0874")
    print(f"Loaded: {dataset.name()}")
    print(f"Train concepts: {len(dataset.train_concepts())} concepts")
    print(f"Test concepts: {len(dataset.test_concepts())} concepts ({len(dataset.test_concepts()[0].data)} samples)")
    print(f"Feature shape: {dataset.train_concepts()[0].data.shape} per concept")
