import numpy as np
import pytest

# ============================================================
# Constants
# ============================================================
from config import * # Import the constants from config.py

# ============================================================
# Function under test
# ============================================================
def orbitalElements2Inertial(r, angles):
    Omega, incl, theta = angles

    # Rotation matrices
    R3_theta = np.array([
        [ np.cos(theta),  np.sin(theta), 0],
        [-np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

    R3_Omega = np.array([
        [ np.cos(Omega),  np.sin(Omega), 0],
        [-np.sin(Omega),  np.cos(Omega), 0],
        [0, 0, 1]
    ])

    R1_inc = np.array([
        [1, 0, 0],
        [0, np.cos(incl), np.sin(incl)],
        [0, -np.sin(incl), np.cos(incl)]
    ])

    perif_to_ECI = (R3_theta @ R1_inc @ R3_Omega).T

    r_perif = np.array([r, 0, 0])
    n = np.sqrt(mu / r**3)
    v_perif = np.array([0, r*n, 0])

    rN = perif_to_ECI @ r_perif
    vN = perif_to_ECI @ v_perif

    return rN, vN


# ============================================================
# Reference Implementation (Ground Truth)
# ============================================================
def oe2rv(r, angles):
    RAAN, incl, theta = angles

    vn = np.sqrt(mu / r)

    # Direct analytical formulas for ECI position and velocity
    # These are known correct for circular orbits.
    r_vec = np.array([
        r * (np.cos(theta) * np.cos(RAAN) - np.sin(theta) * np.cos(incl) * np.sin(RAAN)),
        r * (np.cos(theta) * np.sin(RAAN) + np.sin(theta) * np.cos(incl) * np.cos(RAAN)),
        r * np.sin(theta) * np.sin(incl)
    ])

    v_vec = vn * np.array([
        -np.sin(theta) * np.cos(RAAN) - np.cos(theta) * np.cos(incl) * np.sin(RAAN),
        -np.sin(theta) * np.sin(RAAN) + np.cos(theta) * np.cos(incl) * np.cos(RAAN),
        np.cos(theta) * np.sin(incl)
    ])

    return r_vec, v_vec


# ============================================================
# Utility
# ============================================================
def approx_equal(A, B, tol=1e-8):
    """Utility to compare arrays with tolerance."""
    return np.linalg.norm(np.asarray(A) - np.asarray(B)) <= tol


# ============================================================
# Test 1 — Compare against oe2rv (ground truth)
# Purpose:
#   - These random angle sets test general correctness
#   - Ensures rotation order, signs, and velocity direction match analytical formulas
# ============================================================
@pytest.mark.parametrize("angles", [
    (0.3, 0.5, 1.0),  # moderate RAAN, full inclination, mid anomaly
    (1.2, 0.1, 3.0),  # small inclination case
    (2.5, 1.0, 0.4)   # high RAAN, high inclination
])
def test_comparison_with_analytical(angles):
    r = 7000  # Typical LEO radius; stable, realistic number
    r1, v1 = orbitalElements2Inertial(r, angles)
    r2, v2 = oe2rv(r, angles)

    assert approx_equal(r1, r2)
    assert approx_equal(v1, v2)


# ============================================================
# Test 2 — Special Case: i = 0 (Equatorial orbit)
# Purpose:
#   - When inclination = 0, the orbit lies in the equatorial plane
#   - The rotation matrix simplifies and must match analytical formulas exactly
# ============================================================
def test_special_case_i_0():
    Omega = 1.0   # RAAN is irrelevant when i = 0, but we keep a random value
    incl  = 0.0
    theta = 0.8   # Arbitrary true anomaly to avoid trivial zero-value cases

    rN, vN = orbitalElements2Inertial(7000, (Omega, incl, theta))
    rG, vG = oe2rv(7000, (Omega, incl, theta))

    assert approx_equal(rN, rG)
    assert approx_equal(vN, vG)


# ============================================================
# Test 3 — Special Case: RAAN = 0
# Purpose:
#   - With RAAN = 0, rotation simplifies to only inclination + true anomaly
# ============================================================
def test_special_case_raan_0():
    Omega = 0.0
    incl  = np.deg2rad(45)  # 45° is a clean nontrivial inclination
    theta = np.deg2rad(30)  # 30° gives asymmetric x/y contributions

    rN, vN = orbitalElements2Inertial(7000, (Omega, incl, theta))
    rG, vG = oe2rv(7000, (Omega, incl, theta))

    assert approx_equal(rN, rG)
    assert approx_equal(vN, vG)


# ============================================================
# Test 4 — Orthogonality of r ⋅ v = 0 (Circular orbit)
# Purpose:
#   - For circular orbits, velocity is always perpendicular to position
# ============================================================
def test_orthogonality():
    r = 7000  # typical circular orbit radius
    Omega = 1.0
    incl = np.deg2rad(45)
    theta = np.deg2rad(30)

    rN, vN = orbitalElements2Inertial(r, (Omega, incl, theta))

    dot_rv = np.dot(rN, vN)
    assert abs(dot_rv) < 1e-8


# ============================================================
# Test 5 — Energy & Angular Momentum
# Purpose:
#   - Circular orbit has known specific energy and angular momentum
# ============================================================
def test_energy_and_angular_momentum():
    r = 7000  # typical circular orbit radius
    Omega = 1.0
    incl = np.deg2rad(45)
    theta = np.deg2rad(30)

    rN, vN = orbitalElements2Inertial(r, (Omega, incl, theta))

    # Energy test: specific energy = 0.5 * |v|^2 - mu / r
    energy = 0.5 * np.dot(vN, vN) - mu / np.linalg.norm(rN)
    expected_energy = -mu / (2 * r)
    energy_err = abs(energy - expected_energy)
    assert energy_err < 1e-8

    # Angular momentum test: h = r × v
    h = np.cross(rN, vN)
    expected_h = r * np.sqrt(mu / r)
    h_err = abs(np.linalg.norm(h) - expected_h)
    assert h_err < 1e-8


# ============================================================
# Run all tests with pytest (no need to call explicitly, just run pytest in terminal)
# Example: pytest -v
