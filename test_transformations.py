import numpy as np
import pytest

# -------------------------
# Helpers (same as before)
# -------------------------

def tilde_matrix(v):
    """Skew-symmetric (cross-product) matrix for vector v (3,)"""
    v = np.asarray(v).reshape(3)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])

def normalize_mrp(sigma):
    """Return principal MRP set: if |sigma| > 1 use shadow set"""
    sigma = np.asarray(sigma).astype(float).reshape(3)
    s2 = np.dot(sigma, sigma)
    if s2 > 1.0:
        return -sigma / s2
    return sigma

def normalize_dcm(C):
    """Project C to nearest orthonormal rotation matrix using SVD (numerical stability)"""
    U, _, Vt = np.linalg.svd(C)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def random_unit_quaternion():
    """Generate a random unit quaternion [q0, q1, q2, q3] scalar-first"""
    q = np.random.normal(size=4)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q

def quaternion_to_dcm(q):
    """Quaternion (scalar-first) to DCM."""
    q = np.asarray(q).reshape(4)
    q0, q1, q2, q3 = q
    C = np.array([
        [1 - 2*(q2*q2 + q3*q3),     2*(q1*q2 - q0*q3),         2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),         1 - 2*(q1*q1 + q3*q3),     2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),         2*(q2*q3 + q0*q1),         1 - 2*(q1*q1 + q2*q2)]
    ])
    return C

def axis_angle_to_dcm(axis, angle):
    """Rodrigues formula: axis (3,), angle scalar -> DCM"""
    axis = np.asarray(axis).astype(float).reshape(3)
    axis = axis / np.linalg.norm(axis)
    K = tilde_matrix(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

# -------------------------
# Conversion functions (same as before)
# -------------------------

def MRP2DCM(sigma):
    sigma = normalize_mrp(sigma)
    sigma_squared = np.inner(sigma, sigma)
    S = tilde_matrix(sigma)
    DCM = np.eye(3) + (8 * (S @ S) - 4 * (1 - sigma_squared) * S) / (1 + sigma_squared)**2
    return normalize_dcm(DCM)

def MRP2DCM_(sigma):
    sigma = normalize_mrp(sigma)
    s2 = float(sigma @ sigma)
    S = tilde_matrix(sigma)
    C = np.eye(3) + (8 * (S @ S) - 4 * (1 - s2) * S) / (1 + s2)**2
    return normalize_dcm(C)

def dcm2quat(C):
    C = np.asarray(C).reshape(3, 3)
    tr = np.trace(C)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        q0 = 0.25 * S
        q1 = (C[2, 1] - C[1, 2]) / S
        q2 = (C[0, 2] - C[2, 0]) / S
        q3 = (C[1, 0] - C[0, 1]) / S
    else:
        if (C[0, 0] > C[1, 1]) and (C[0, 0] > C[2, 2]):
            S = np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2]) * 2.0
            q0 = (C[2, 1] - C[1, 2]) / S
            q1 = 0.25 * S
            q2 = (C[0, 1] + C[1, 0]) / S
            q3 = (C[0, 2] + C[2, 0]) / S
        elif C[1, 1] > C[2, 2]:
            S = np.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2]) * 2.0
            q0 = (C[0, 2] - C[2, 0]) / S
            q1 = (C[0, 1] + C[1, 0]) / S
            q2 = 0.25 * S
            q3 = (C[1, 2] + C[2, 1]) / S
        else:
            S = np.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1]) * 2.0
            q0 = (C[1, 0] - C[0, 1]) / S
            q1 = (C[0, 2] + C[2, 0]) / S
            q2 = (C[1, 2] + C[2, 1]) / S
            q3 = 0.25 * S

    q = np.array([q0, q1, q2, q3])
    if q[0] < 0:
        q = -q
    q /= np.linalg.norm(q)
    return q

def DCM2MRP(matrix):
    C = np.asarray(matrix).reshape(3, 3)
    C = normalize_dcm(C)
    trace = np.trace(C)
    zeta = np.sqrt(max(trace + 1.0, 0.0))
    denom = (zeta**2 + 2.0 * zeta)
    if denom < 1e-16:
        diag = np.diag(C)
        k = int(np.argmax(diag))
        sigma = np.zeros(3)
        sigma[k] = np.sqrt(max((diag[k] + 1.0) / 2.0, 0.0))
        return normalize_mrp(sigma)
    s1 = (C[1, 2] - C[2, 1]) / denom
    s2 = (C[2, 0] - C[0, 2]) / denom
    s3 = (C[0, 1] - C[1, 0]) / denom
    sigma = np.array([s1, s2, s3])
    return normalize_mrp(sigma)

def DCM2MRP_(C):
    C = np.asarray(C).reshape(3, 3)
    b = dcm2quat(C)
    if abs(1.0 + b[0]) < 1e-12:
        return DCM2MRP(C)
    q = np.zeros(3)
    q[0] = b[1] / (1.0 + b[0])
    q[1] = b[2] / (1.0 + b[0])
    q[2] = b[3] / (1.0 + b[0])
    return normalize_mrp(q)

# -------------------------
# Test functions (converted to pytest)
# -------------------------

def approx_equal(A, B, tol=1e-9):
    """Helper function to compare two matrices/vectors for equality with a given tolerance."""
    return np.linalg.norm(np.asarray(A) - np.asarray(B)) <= tol

def test_identity():
    """Test for identity behavior:
    1. Starting with a zero MRP (sigma0), it should generate an identity rotation matrix (C).
    2. The quaternion generated from this matrix should be [1.0, 0.0, 0.0, 0.0] (indicating no rotation).
    3. The inverse transformation from DCM to MRP should return the original sigma0 vector.
    """
    sigma0 = np.zeros(3)
    C = MRP2DCM(sigma0)
    q = dcm2quat(C)
    sigma_back = DCM2MRP(C)
    assert approx_equal(C, np.eye(3), tol=1e-12)  # Identity matrix
    assert approx_equal(q, np.array([1.0, 0.0, 0.0, 0.0]), tol=1e-12)  # Quaternion identity
    assert approx_equal(sigma_back, sigma0, tol=1e-12)  # Original MRP


def test_round_trip_mrp_dcm(num_samples=200):
    """Test the round-trip conversion from MRP to DCM and back to MRP.
    1. Random MRPs are converted to DCMs.
    2. The DCM is then converted back to MRP and should return to the original value within a tolerance.
    This ensures the transformations are consistent.
    """
    for i in range(num_samples):
        axis = np.random.normal(size=3)
        axis /= np.linalg.norm(axis)
        mag = np.random.uniform(0.0, 1.6)
        sigma = axis * mag
        C = MRP2DCM(sigma)
        sigma2 = DCM2MRP(C)
        C2 = MRP2DCM(sigma2)
        assert approx_equal(C, C2, tol=5e-10)  # Should match original C


def test_dcm_quat_mrp_consistency(num_samples=200):
    """Test the consistency between DCM, quaternion, and MRP.
    1. Random unit quaternions are converted to DCM.
    2. The DCM is converted back to quaternion and MRP.
    3. The original DCM should match the one computed from the MRP.
    """
    for i in range(num_samples):
        q = random_unit_quaternion()
        C = quaternion_to_dcm(q)
        C = normalize_dcm(C)
        q_from_C = dcm2quat(C)
        sigma_from_C = DCM2MRP_(C)
        C_back = MRP2DCM(sigma_from_C)
        assert approx_equal(C, C_back, tol=5e-10)  # Should match original C


def test_180_degree_cases():
    """Test for 180-degree axis-angle rotations.
    1. Rotations of 180 degrees around standard axes (x, y, z) should result in specific behaviors.
    2. Random axes are tested to ensure that the conversion from DCM to MRP and back is correct.
    3. If the test fails for any axis, an assertion error is raised.
    """
    axes = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    axes += [np.random.normal(size=3) for _ in range(3)]
    for i, a in enumerate(axes):
        a = a / np.linalg.norm(a)
        C = axis_angle_to_dcm(a, np.pi)  # 180 degrees (pi radians)
        try:
            sigma = DCM2MRP(C)
            C_back = MRP2DCM(sigma)
            assert approx_equal(C, C_back, tol=1e-9)  # Should match original C
        except Exception:
            assert False, f"180° axis {i} failed"


def test_shadow_set_behavior():
    """Test behavior for MRP values with norm > 1.
    1. MRP values with a norm greater than 1 should automatically be shadowed (normalized).
    2. The resulting MRP should have a norm <= 1.
    """
    sigma = np.array([2.0, 0.3, -0.5])  # norm > 1
    C = MRP2DCM(sigma)
    sigma_converted = DCM2MRP(C)
    assert np.linalg.norm(sigma_converted) <= 1.0 + 1e-12  # Should normalize to shadow set


def test_numerical_properties_random(num_samples=500):
    """Test numerical properties of random DCMs.
    1. Small random perturbations are applied to DCM matrices.
    2. The resulting matrix should still be orthonormal (i.e., its transpose multiplied by itself should be the identity).
    3. The determinant of the matrix should be close to 1 (determinant of rotation matrices is always 1).
    """
    for i in range(num_samples):
        q = random_unit_quaternion()
        C = quaternion_to_dcm(q)
        C_pert = C + 1e-12 * np.random.normal(size=(3, 3))  # Apply small perturbation
        C_proj = normalize_dcm(C_pert)
        ortho_err = np.linalg.norm(C_proj.T @ C_proj - np.eye(3))  # Check orthogonality
        det_err = abs(np.linalg.det(C_proj) - 1.0)  # Check determinant
        assert ortho_err <= 1e-12
        assert det_err <= 1e-12


# If you want to run the tests, execute this command in the terminal:
# pytest -v
