from typing import Any, Dict, Optional


def human_decision_fn(seg_id: int, segment_info: Optional[Dict[str, Any]] = None) -> int:
    """
    Parameters
    ----------
    seg_id : int
        Unique telemetry segment identifier.
    segment_info : Optional[Dict[str, Any]]

    Returns
    -------
    int
        0 for verified normal (B_N), 1 for verified anomaly (B_A).
    """
    expert_annotations: Dict[int, int] = {
        
    }

    if seg_id in expert_annotations:
        return expert_annotations[seg_id]

    if segment_info and "ground_truth" in segment_info:
        return int(segment_info["ground_truth"])

    return 0
