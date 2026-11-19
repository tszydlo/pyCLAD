import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def format_results_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Format evaluation metrics into a clean summary DataFrame.
    """
    return pd.DataFrame(results)


def plot_hitl_benchmark(
    df: pd.DataFrame,
    output_path: Optional[Union[str, pathlib.Path]] = "hitl_benchmark.png",
    show: bool = False,
    save_path: Optional[Union[str, pathlib.Path]] = None,
) -> Tuple[plt.Figure, Any]:
    """
    Plot master grouped bar chart comparing algorithms across budgets and anomaly ratios.
    """
    target_path = output_path if output_path is not None else save_path

    ratios = [0, 10, 50]
    budgets = [10, 25, 50, 75]
    budget_colors = ["#42A5F5", "#1E88E5", "#1565C0", "#0D47A1"]

    algos = df["Algorithm"].unique()
    fig, axes = plt.subplots(len(ratios), 1, figsize=(10, 11), sharex=True, sharey=True)
    if len(ratios) == 1:
        axes = [axes]

    bar_width = 0.18
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, len(budgets))
    x = np.arange(len(algos))

    for idx, r in enumerate(ratios):
        ax = axes[idx]
        mean_floor = df["Floor"].mean()
        mean_ceiling = df["Ceiling"].mean()

        ax.axhline(
            mean_ceiling, color="#1B5E20", linestyle="--", linewidth=2.5, label=f"RF Ceiling ({mean_ceiling:.3f})"
        )
        ax.axhline(mean_floor, color="#D32F2F", linestyle="--", linewidth=2.0, label=f"Unsup Floor ({mean_floor:.3f})")

        for a_idx, algo in enumerate(algos):
            algo_row = df[df["Algorithm"] == algo].iloc[0]
            for b_idx, budget in enumerate(budgets):
                col_name = f"BA{budget},{r}"
                val = algo_row.get(col_name, 0.5)
                pos = a_idx + offsets[b_idx]
                ax.bar(pos, val, bar_width, color=budget_colors[b_idx], edgecolor="#212121", linewidth=0.8, alpha=0.9)
                ax.text(
                    pos,
                    0.05,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                    color="white",
                    rotation=90,
                )

        ax.set_title(f"Calibration Anomaly Ratio: {r}%", fontsize=11, fontweight="bold")
        ax.set_ylabel("Balanced Accuracy", fontsize=10, fontweight="bold")
        ax.set_ylim(0.0, 1.10)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(algos, fontsize=10, fontweight="bold")
    axes[0].legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    if target_path:
        path = pathlib.Path(target_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig, axes
