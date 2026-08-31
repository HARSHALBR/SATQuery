"""
Unit tests for Sentinel-1 (SAR) and Sentinel-2 (Multispectral) pairing and co-registration checks.
"""

import pytest
from data.bigearthnet_txt.utils import validate_s1_s2_pairing


def test_valid_pairing():
    s1_name = "S1A_IW_GRDH_1SDV_20170613T165043_33N_53E_6367f0"
    s2_name = "S2A_MSIL2A_20170613T101031_N0205_R022_T32ULD_22_38"
    valid, err = validate_s1_s2_pairing(s1_name, s2_name)
    assert valid is True
    assert err is None


def test_invalid_pairing_empty_names():
    valid, err = validate_s1_s2_pairing("", "S2A_MSIL2A_20170613T101031")
    assert valid is False
    assert "Empty" in err

    valid, err = validate_s1_s2_pairing("S1A_IW_GRDH", "")
    assert valid is False
    assert "Empty" in err


def test_invalid_prefix_convention():
    valid, err = validate_s1_s2_pairing("Landsat8_OLI_123", "S2A_MSIL2A_123")
    assert valid is False
    assert "Invalid S1" in err

    valid, err = validate_s1_s2_pairing("S1A_IW_GRDH_123", "MODIS_Terra_123")
    assert valid is False
    assert "Invalid S2" in err


def test_pairing_spatial_distance_check():
    s1_name = "S1A_IW_GRDH_1SDV_20170613T165043_33N_53E_6367f0"
    s2_name = "S2A_MSIL2A_20170613T101031_N0205_R022_T32ULD_22_38"

    # Close coordinates -> Valid
    s1_meta = {"latitude": 45.123, "longitude": 12.456}
    s2_meta = {"latitude": 45.125, "longitude": 12.458}
    valid, err = validate_s1_s2_pairing(s1_name, s2_name, s1_meta, s2_meta)
    assert valid is True

    # Far coordinates -> Invalid (spatial mismatch)
    s1_far = {"latitude": 45.123, "longitude": 12.456}
    s2_far = {"latitude": 55.999, "longitude": 24.123}
    valid, err = validate_s1_s2_pairing(s1_name, s2_name, s1_far, s2_far)
    assert valid is False
    assert "Spatial distance too large" in err
