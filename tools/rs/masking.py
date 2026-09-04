import numpy as np

# SCL Classes
# 0: No Data
# 1: Saturated or defective
# 2: Dark Area Pixels
# 3: Cloud Shadows
# 4: Vegetation
# 5: Not Vegetated
# 6: Water
# 7: Unclassified
# 8: Cloud Medium Probability
# 9: Cloud High Probability
# 10: Thin Cirrus
# 11: Snow / Ice

INVALID_SCL_CLASSES = {0, 1, 3, 7, 8, 9, 10, 11}
# We optionally allow 2 (Dark areas) but exclude 3 (Cloud Shadows).
# 6 (Water) is valid for NDVI (will just be negative).

def create_valid_mask(scl_array: np.ndarray, extra_invalid_classes=None) -> np.ndarray:
    """
    Creates a boolean mask where True indicates a valid pixel, False indicates invalid.
    """
    invalid_classes = set(INVALID_SCL_CLASSES)
    if extra_invalid_classes:
        invalid_classes.update(extra_invalid_classes)
        
    mask = np.ones(scl_array.shape, dtype=bool)
    for c in invalid_classes:
        mask &= (scl_array != c)
        
    return mask

def combined_valid_mask(t1_scl: np.ndarray, t2_scl: np.ndarray) -> np.ndarray:
    """
    Combines valid masks for T1 and T2 (logical AND).
    """
    mask1 = create_valid_mask(t1_scl)
    mask2 = create_valid_mask(t2_scl)
    return mask1 & mask2
