from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from pyclad.analysis.benchmark_reporting import (
    format_results_table,
    plot_hitl_benchmark,
)
from pyclad.models.adapters.adaptive_threshold import AdaptiveThresholdDetector
from pyclad.strategies.replay.buffers.hitl_buffers import DualLabelBuffer
from pyclad.strategies.replay.hitl_feedback import human_decision_fn


def populate_dual_buffer_with_budget(
    train_pool: pd.DataFrame,
    features: List[str],
    budget: float,
    anomaly_ratio: float,
    seed: int = 42,
) -> DualLabelBuffer:
    """
    Populate DualLabelBuffer using ratio-constrained random budget selection.
    """
    np.random.seed(seed)
    X = train_pool[features].values.astype(np.float64)
    y = train_pool["anomaly"].values.astype(int)

    norm_idx = np.where(y == 0)[0]

    n_prov = int(round(len(train_pool) * budget))
    prov_idx = np.random.choice(len(train_pool), size=n_prov, replace=False)
    rem_idx = np.setdiff1d(np.arange(len(train_pool)), prov_idx)

    prov_norm = [i for i in prov_idx if y[i] == 0]
    prov_anom = [i for i in prov_idx if y[i] == 1]
    rem_norm = [i for i in rem_idx if y[i] == 0]
    rem_anom = [i for i in rem_idx if y[i] == 1]

    if anomaly_ratio == 0.0:
        n_target_anom = 0
        n_target_norm = len(norm_idx)
    else:
        anom_needed = max(1, int(round(len(norm_idx) * anomaly_ratio / (1.0 - anomaly_ratio))))
        tot_anom = len(prov_anom) + len(rem_anom)
        if tot_anom >= anom_needed:
            n_target_norm = len(norm_idx)
            n_target_anom = anom_needed
        else:
            n_target_anom = tot_anom
            n_target_norm = min(
                len(norm_idx), max(1, int(round(n_target_anom * (1.0 - anomaly_ratio) / anomaly_ratio)))
            )

    n_init_norm = min(len(prov_norm), n_target_norm)
    sel_prov_norm = (
        np.random.choice(prov_norm, size=n_init_norm, replace=False) if n_init_norm > 0 else np.array([], dtype=int)
    )
    extra_norm = n_target_norm - n_init_norm
    sel_rem_norm = (
        np.random.choice(rem_norm, size=min(len(rem_norm), extra_norm), replace=False)
        if extra_norm > 0
        else np.array([], dtype=int)
    )

    if n_target_anom == 0:
        sel_prov_anom = np.array([], dtype=int)
        sel_rem_anom = np.array([], dtype=int)
    else:
        n_init_anom = min(len(prov_anom), n_target_anom)
        sel_prov_anom = (
            np.random.choice(prov_anom, size=n_init_anom, replace=False) if n_init_anom > 0 else np.array([], dtype=int)
        )
        extra_anom = n_target_anom - n_init_anom
        sel_rem_anom = (
            np.random.choice(rem_anom, size=min(len(rem_anom), extra_anom), replace=False)
            if extra_anom > 0
            else np.array([], dtype=int)
        )

    dual_buf = DualLabelBuffer()
    for idx in sel_prov_norm:
        dual_buf.add(X[idx], label=0)
    for idx in sel_prov_anom:
        dual_buf.add(X[idx], label=1)

    for idx in sel_rem_norm:
        seg_id = int(train_pool.iloc[idx]["segment"])
        dual_buf.add(X[idx], label=human_decision_fn(seg_id, {"ground_truth": y[idx]}))
    for idx in sel_rem_anom:
        seg_id = int(train_pool.iloc[idx]["segment"])
        dual_buf.add(X[idx], label=human_decision_fn(seg_id, {"ground_truth": y[idx]}))

    return dual_buf


def run_pipeline(
    channel: str = "CADC0874",
    output_path: Optional[Union[str, Path]] = "hitl_benchmark.png",
    show: bool = False,
) -> pd.DataFrame:
    """
    Run full benchmark on the selected telemetry channel and display results.
    """
    # Load telemetry data
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "dataset" / "dataset.csv",
        Path("dataset/dataset.csv"),
        Path("pyCLAD-main/dataset/dataset.csv"),
    ]
    data_path = next((p for p in candidates if p.exists()), candidates[0])
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset.csv in '{data_path}'.")

    df = pd.read_csv(data_path)
    features = [c for c in df.columns if c not in ["segment", "anomaly", "train", "channel"]]

    ch_df = df[df["channel"] == channel].sort_values("segment").reset_index(drop=True)
    tr_pool = ch_df[ch_df["train"] == 1].reset_index(drop=True)
    te_pool = ch_df[ch_df["train"] == 0].reset_index(drop=True)

    X_train = tr_pool[features].values.astype(np.float64)
    y_train = tr_pool["anomaly"].values.astype(int)
    X_test = te_pool[features].values.astype(np.float64)
    y_test = te_pool["anomaly"].values.astype(int)

    # Establish Baselines: Floor (Unsupervised IF) & Ceiling (Supervised RF)
    raw_if = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    raw_if.fit(X_train[y_train == 0])
    floor_score = float(balanced_accuracy_score(y_test, (raw_if.predict(X_test) == -1).astype(int)))

    rf_ceil = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    rf_ceil.fit(X_train, y_train)
    ceiling_score = float(balanced_accuracy_score(y_test, rf_ceil.predict(X_test)))

    print(f"Channel: {channel} | Floor: {floor_score:.4f} | Ceiling: {ceiling_score:.4f}")

    # Evaluate Across Labeling Budgets and Anomaly Ratios
    budgets = [0.10, 0.25, 0.50, 0.75]
    ratios = [0.0, 0.10, 0.50]
    record: Dict[str, Any] = {
        "Algorithm": "Adaptive_iForest",
        "Channel": channel,
        "Floor": floor_score,
        "Ceiling": ceiling_score,
    }

    for r in ratios:
        for b in budgets:
            b_key = f"BA{int(b * 100)},{int(r * 100)}"
            dual_buf = populate_dual_buffer_with_budget(tr_pool, features, budget=b, anomaly_ratio=r)

            model = AdaptiveThresholdDetector(random_state=42)
            model.fit(dual_buf.normal_buffer.data())
            model.calibrate(dual_buf.x_labeled, dual_buf.y_labeled)

            preds, _ = model.predict(X_test)
            record[b_key] = float(balanced_accuracy_score(y_test, preds))

    # Format Results & Plot
    res_df = format_results_table([record])
    print("\n Benchmark Results Summary")
    print(res_df.to_string(index=False))

    if output_path:
        plot_hitl_benchmark(res_df, output_path=output_path, show=show)

    print("\nBenchmark completed successfully.")
    return res_df


if __name__ == "__main__":
    run_pipeline()
