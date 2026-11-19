"""Dual Replay Buffers for Human-in-the-Loop Continual Anomaly Detection."""

from typing import Any, Dict, List

import numpy as np

from pyclad.strategies.replay.buffers.buffer import ReplayBuffer


class NormalReplayBuffer(ReplayBuffer):
    """Replay buffer storing verified normal telemetry segments (B_N)."""

    def __init__(self) -> None:
        self._buffers: List[np.ndarray] = []

    def update(self, data: np.ndarray) -> None:
        arr = np.asarray(data, dtype=np.float64)
        self._buffers.append(arr.reshape(-1, arr.shape[-1] if arr.ndim > 1 else len(arr)))

    def data(self) -> np.ndarray:
        return np.vstack(self._buffers) if self._buffers else np.empty((0, 0))

    def name(self) -> str:
        return "NormalReplayBuffer"

    def additional_info(self) -> Dict[str, Any]:
        return {"current_size": len(self._buffers)}


class AnomalousReplayBuffer(ReplayBuffer):
    """Replay buffer storing verified anomalous telemetry segments (B_A)."""

    def __init__(self) -> None:
        self._buffers: List[np.ndarray] = []

    def update(self, data: np.ndarray) -> None:
        arr = np.asarray(data, dtype=np.float64)
        self._buffers.append(arr.reshape(-1, arr.shape[-1] if arr.ndim > 1 else len(arr)))

    def data(self) -> np.ndarray:
        return np.vstack(self._buffers) if self._buffers else np.empty((0, 0))

    def name(self) -> str:
        return "AnomalousReplayBuffer"

    def additional_info(self) -> Dict[str, Any]:
        return {"current_size": len(self._buffers)}


class DualLabelBuffer:
    """Coordinates NormalReplayBuffer and AnomalousReplayBuffer for HITL calibration."""

    def __init__(self) -> None:
        self.normal_buffer = NormalReplayBuffer()
        self.anomalous_buffer = AnomalousReplayBuffer()

    def add(self, data: np.ndarray, label: int) -> None:
        if label == 0:
            self.normal_buffer.update(data)
        else:
            self.anomalous_buffer.update(data)

    @property
    def x_labeled(self) -> np.ndarray:
        """Return stacked feature matrix of all labeled segments."""
        n_d, a_d = self.normal_buffer.data(), self.anomalous_buffer.data()
        if len(n_d) == 0:
            return a_d
        if len(a_d) == 0:
            return n_d
        return np.vstack([n_d, a_d])

    @property
    def y_labeled(self) -> np.ndarray:
        """Return combined binary ground-truth labels (0 for normal, 1 for anomalous)."""
        return np.concatenate(
            [
                np.zeros(len(self.normal_buffer.data()), dtype=int),
                np.ones(len(self.anomalous_buffer.data()), dtype=int),
            ]
        )

    @property
    def normal_count(self) -> int:
        return len(self.normal_buffer.data())

    @property
    def anomalous_count(self) -> int:
        return len(self.anomalous_buffer.data())

    @property
    def total_count(self) -> int:
        return self.normal_count + self.anomalous_count
