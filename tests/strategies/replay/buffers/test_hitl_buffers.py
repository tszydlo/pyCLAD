import numpy as np

from pyclad.strategies.replay.buffers.buffer import ReplayBuffer
from pyclad.strategies.replay.buffers.hitl_buffers import (
    AnomalousReplayBuffer,
    DualLabelBuffer,
    NormalReplayBuffer,
)


def test_buffers_inheritance_and_storage():
    normal_buf = NormalReplayBuffer()
    anom_buf = AnomalousReplayBuffer()

    assert isinstance(normal_buf, ReplayBuffer)
    assert isinstance(anom_buf, ReplayBuffer)

    normal_buf.update(np.ones((5, 10)))
    anom_buf.update(np.zeros((3, 10)))

    assert normal_buf.data().shape == (5, 10)
    assert anom_buf.data().shape == (3, 10)
    assert normal_buf.info()["buffer_size"] == 5
    assert anom_buf.info()["buffer_size"] == 3


def test_dual_label_buffer_routing():
    dual = DualLabelBuffer()
    dual.add(np.ones(10), label=0)
    dual.add(np.zeros(10), label=1)

    assert dual.normal_count == 1
    assert dual.anomalous_count == 1
    assert dual.total_count == 2
    assert len(dual.x_labeled) == 2
    assert np.array_equal(dual.y_labeled, np.array([0, 1]))
