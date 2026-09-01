import numpy as np

def compute_change_statistics(delta_map: np.ndarray, valid_mask: np.ndarray, threshold: float, change_type: str = "decrease") -> dict:
    """
    Computes change statistics over a delta map.
    If change_type is 'decrease', checks delta <= threshold.
    If change_type is 'increase', checks delta >= threshold.
    Returns:
        total_valid_pixels: int
        decrease_pixel_fraction: float
        increase_pixel_fraction: float
        mean_delta: float
        threshold_used: float
        change_mask: np.ndarray
    """
    if not np.any(valid_mask):
        return {
            "total_valid_pixels": 0,
            "decrease_pixel_fraction": 0.0,
            "increase_pixel_fraction": 0.0,
            "mean_delta": 0.0,
            "threshold_used": threshold,
            "change_mask": np.zeros_like(delta_map, dtype=bool)
        }
        
    valid_deltas = delta_map[valid_mask]
    total_valid = len(valid_deltas)
    
    decrease_mask = (valid_deltas <= threshold) if threshold < 0 else (valid_deltas <= -threshold)
    # Using negative threshold explicitly for decrease, or assuming threshold passed is the cut-off.
    # The instruction says: "If threshold represents a decrease threshold: decrease if delta <= threshold".
    # I'll use <= threshold for decrease, and >= abs(threshold) for increase.
    # Actually, we should just measure both. Let's assume threshold is the magnitude (e.g. -0.15 for decrease).
    
    dec_thresh = threshold if threshold < 0 else -threshold
    inc_thresh = threshold if threshold > 0 else -threshold
    
    dec_pixels = np.sum(valid_deltas <= dec_thresh)
    inc_pixels = np.sum(valid_deltas >= inc_thresh)
    
    mean_delta = float(np.mean(valid_deltas))
    
    full_change_mask = np.zeros_like(delta_map, dtype=bool)
    if change_type == "decrease":
        full_change_mask[valid_mask] = (valid_deltas <= dec_thresh)
    else:
        full_change_mask[valid_mask] = (valid_deltas >= inc_thresh)
        
    return {
        "total_valid_pixels": int(total_valid),
        "decrease_pixel_fraction": float(dec_pixels / total_valid),
        "increase_pixel_fraction": float(inc_pixels / total_valid),
        "mean_delta": mean_delta,
        "threshold_used": dec_thresh if change_type == "decrease" else inc_thresh,
        "change_mask": full_change_mask
    }
