import numpy as np

def compute_ndvi(red: np.ndarray, nir: np.ndarray, valid_mask: np.ndarray = None) -> np.ndarray:
    """
    Computes NDVI = (NIR - RED) / (NIR + RED).
    Reflectance scaling is invariant (e.g., both divided by 10000) so we compute on raw arrays.
    Returns NaN where invalid, denom is zero, or masked.
    """
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    
    denom = nir + red
    num = nir - red
    
    # Avoid division by zero
    safe_denom = np.where(denom == 0, 1.0, denom)
    ndvi = num / safe_denom
    
    # Apply conditions: denom == 0 or explicit mask
    invalid = (denom == 0)
    if valid_mask is not None:
        invalid |= (~valid_mask)
        
    ndvi[invalid] = np.nan
    return ndvi

def compute_ndvi_delta(t1_red: np.ndarray, t1_nir: np.ndarray, 
                       t2_red: np.ndarray, t2_nir: np.ndarray, 
                       valid_mask: np.ndarray = None) -> tuple:
    """
    Computes delta NDVI = NDVI_T2 - NDVI_T1.
    A pixel is valid only if both dates are valid.
    Returns: (ndvi_t1, ndvi_t2, delta_ndvi, final_valid_mask)
    """
    ndvi_t1 = compute_ndvi(t1_red, t1_nir, valid_mask)
    ndvi_t2 = compute_ndvi(t2_red, t2_nir, valid_mask)
    
    delta = ndvi_t2 - ndvi_t1
    
    # Final validity: not NaN in delta
    final_valid_mask = ~np.isnan(delta)
    if valid_mask is not None:
        final_valid_mask &= valid_mask
        
    return ndvi_t1, ndvi_t2, delta, final_valid_mask
