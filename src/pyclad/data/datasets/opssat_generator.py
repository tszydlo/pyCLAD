"""
OPS-SAT Data Generator.
Transforms raw telemetry segments into continual learning concepts.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

OPS_SAT_FEATURES = [
    "mean",
    "var",
    "std",
    "len",
    "duration",
    "len_weighted",
    "gaps_squared",
    "n_peaks",
    "smooth10_n_peaks",
    "smooth20_n_peaks",
    "var_div_duration",
    "var_div_len",
    "diff_peaks",
    "diff2_peaks",
    "diff_var",
    "diff2_var",
    "kurtosis",
    "skew",
]

OPS_SAT_CHANNELS = [
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

TARGET_ANOMALY_RATIO = 0.20


def make_synthetic_anomalies(normal_data: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    means = normal_data.mean(axis=0)
    stds = normal_data.std(axis=0)
    stds = np.where(stds == 0, 1e-9, stds)

    points = []
    for _ in range(n):
        k = rng.uniform(3.0, 5.0, size=normal_data.shape[1])
        direction = rng.choice([-1, 1], size=normal_data.shape[1])
        points.append(means + direction * k * stds)
    return np.array(points)


def make_synthetic_normals(normal_data: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    means = normal_data.mean(axis=0)
    stds = normal_data.std(axis=0)
    stds = np.where(stds == 0, 1e-9, stds)
    return rng.normal(loc=means, scale=stds * 0.5, size=(n, normal_data.shape[1]))


def generate_ops_sat_concepts(
    raw_df: pd.DataFrame,
    channel: str,
    feature_cols: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if feature_cols is None:
        feature_cols = OPS_SAT_FEATURES

    rng = np.random.default_rng(seed)
    ch_df = raw_df[raw_df["channel"] == channel].sort_values("segment").reset_index(drop=True)
    if ch_df.empty:
        raise ValueError(f"Channel '{channel}' not found in raw dataset.")

    ch_df = ch_df.dropna(subset=feature_cols).copy()

    # Train: nominal segments only
    train_rows_raw = ch_df[(ch_df["train"] == 1)]# & (ch_df["anomaly"] == 0)]
    train_X = train_rows_raw[feature_cols].values.astype(np.float64)
    train_labels = train_rows_raw["anomaly"].values.astype(int)

    if len(train_X) == 0:
        raise ValueError(f"No nominal training data for channel {channel}")

    train_rows = []
    for i in range(len(train_X)):
        row_dict = {
            "concept_id": i,
            "concept_name": f"{channel}_train_{i:04d}",
            "label": int(train_labels[i]),
        }
        for f_idx, f in enumerate(feature_cols):
            row_dict[f] = float(train_X[i, f_idx])
        train_rows.append(row_dict)
    train_df = pd.DataFrame(train_rows)

    # Test: nominal + anomalous segments
    test_rows_raw = ch_df[ch_df["train"] == 0].sort_values("segment").reset_index(drop=True)
    test_X = test_rows_raw[feature_cols].values.astype(np.float64)
    test_labels = test_rows_raw["anomaly"].values.astype(int)

    # If no anomalies in test, inject synthetic ones
    if test_labels.sum() == 0:
        n_syn = max(1, round(len(test_labels) * TARGET_ANOMALY_RATIO / (1 - TARGET_ANOMALY_RATIO)))
        syn_X = make_synthetic_anomalies(train_X, n_syn, rng)
        test_X = np.vstack([test_X, syn_X])
        test_labels = np.concatenate([test_labels, np.ones(n_syn, dtype=int)])

    # If no normals in test, inject synthetic ones
    if (test_labels == 0).sum() == 0:
        n_anom = (test_labels == 1).sum()
        n_syn = max(1, round(n_anom * (1 - TARGET_ANOMALY_RATIO) / TARGET_ANOMALY_RATIO))
        syn_X = make_synthetic_normals(train_X, n_syn, rng)
        test_X = np.vstack([test_X, syn_X])
        test_labels = np.concatenate([test_labels, np.zeros(n_syn, dtype=int)])

    test_rows = []
    for i in range(len(test_X)):
        row_dict = {
            "concept_id": 0,
            "concept_name": f"{channel}_test_full",
            "label": int(test_labels[i]),
        }
        for f_idx, f in enumerate(feature_cols):
            row_dict[f] = float(test_X[i, f_idx])
        test_rows.append(row_dict)
    test_df = pd.DataFrame(test_rows)

    return train_df, test_df


def process_and_save_all_channels(
    raw_csv_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Tuple[int, int]]:
    """
    Processes all 9 channels from raw dataset.csv and writes concept CSVs to output_dir.
    """
    if raw_csv_path is None:
        candidates = [
            Path("dataset/dataset.csv"),
            Path("pyCLAD-main/dataset/dataset.csv"),
            Path(__file__).resolve().parents[4] / "dataset" / "dataset.csv",
        ]
        raw_csv_path = next((p for p in candidates if p.exists()), Path("dataset/dataset.csv"))
    else:
        raw_csv_path = Path(raw_csv_path)

    if output_dir is None:
        output_dir = raw_csv_path.parent / "opssat_lad_data"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(raw_csv_path)

    stats = {}
    for ch in OPS_SAT_CHANNELS:
        train_df, test_df = generate_ops_sat_concepts(raw_df, ch)
        train_df.to_csv(output_dir / f"{ch}_train.csv", index=False)
        test_df.to_csv(output_dir / f"{ch}_test.csv", index=False)
        stats[ch] = (len(train_df), len(test_df))
        train_normal = int((train_df["label"] == 0).sum())
        train_anomaly = int((train_df["label"] == 1).sum())
        print(
            f"Channel {ch:<10} -> Train Concepts: {len(train_df):>3} "
            f"(Normal: {train_normal:>3}, Anomaly: {train_anomaly:>3}) | Test Samples: {len(test_df):>3}"
        )
    return stats


if __name__ == "__main__":
    process_and_save_all_channels()
