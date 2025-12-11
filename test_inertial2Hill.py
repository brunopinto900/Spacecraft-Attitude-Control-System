import numpy as np
import pytest
from config import * # Import the constants from config.py



# Example function to normalize a DCM
def normalize_dcm(R):
    """Perform SVD to re-orthogonalize the matrix."""
    U, _, Vt = np.linalg.svd(R)
    R_normalized = np.dot(U, Vt)
    return R_normalized


def orbitalElements2Inertial(r, angles):
    Omega, incl, theta = angles

    # Rotation matrices
    R3_theta = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    R3_Omega = np.array([
        [np.cos(Omega), np.sin(Omega), 0],
        [-np.sin(Omega), np.cos(Omega), 0],
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
    v_perif = np.array([0, r * n, 0])

    rN = perif_to_ECI @ r_perif
    vN = perif_to_ECI @ v_perif

    return rN, vN


def Inertial2Hill(t):
    '''
    Computes the Direct Cosine Matrix from the Hill Frame (LMO) to the inertial frame (N)
    Arguments:
        t {float} -- time from t_0, in seconds
    Returns:
        (3,3) np.array -- Direct Cosine Matrix HN
    '''
    # Initial conditions for LMO
    theta_dot_LMO = np.sqrt(mu / r_LMO**3)  # rad/sec
    angles_LMO = np.array([20, 30, 60]) * np.pi / 180  # Ω,i,θ(t_0), in radians
    angles_LMO += np.array([0, 0, theta_dot_LMO * t])  # Ω,i,θ(t), in radians
    
    # Computing inertial np.arrays
    r_inertial_LMO, r_dot_inertial_LMO = orbitalElements2Inertial(r_LMO, angles_LMO)
    
    # Computing Hill Frame versors
    i_r = r_inertial_LMO / np.linalg.norm(r_inertial_LMO)
    i_h = np.cross(r_inertial_LMO, r_dot_inertial_LMO) / np.linalg.norm(np.cross(r_inertial_LMO, r_dot_inertial_LMO))
    i_theta = np.cross(i_h, i_r)
    NH = np.array([i_r, i_theta, i_h])
    HN = NH.T
    return normalize_dcm(HN)


# Test functions converted to pytest

def approx_equal(A, B, tol=1e-10):
    """Utility to compare arrays with tolerance."""
    return np.linalg.norm(np.asarray(A) - np.asarray(B)) <= tol


def test_orthonormality():
    """Test that the Direct Cosine Matrix (HN) is orthonormal.
    1. The matrix HN should satisfy the property HN^T * HN = Identity.
    2. The determinant of HN should be +1, confirming it represents a valid rotation.
    """
    t = 500  # arbitrary nonzero time
    HN = Inertial2Hill(t)

    # HN^T * HN should be Identity
    I_test = HN.T @ HN
    I = np.eye(3)
    assert approx_equal(I_test, I)

    # determinant should be +1
    det = np.linalg.det(HN)
    assert abs(det - 1) < 1e-10


def test_axes_definitions():
    """Test that the axes defined in the Hill frame correspond to the expected inertial axes.
    1. Computes the position and velocity in the inertial frame using orbital elements.
    2. Extracts the Hill frame vectors from the Direct Cosine Matrix (HN).
    3. The Hill axes (i_r, i_theta, i_h) should match the expected inertial directions.
    """
    t = 200
    HN = Inertial2Hill(t)

    # Compute inertial pos/vel manually
    theta_dot_LMO = np.sqrt(mu / r_LMO**3)
    angles = np.array([20, 30, 60]) * np.pi / 180
    angles += np.array([0, 0, theta_dot_LMO * t])

    rN, vN = orbitalElements2Inertial(r_LMO, angles)

    # Analytical Hill directions
    i_r_expected = rN / np.linalg.norm(rN)
    i_h_expected = np.cross(rN, vN)
    i_h_expected /= np.linalg.norm(i_h_expected)
    i_theta_expected = np.cross(i_h_expected, i_r_expected)

    # Extract vectors from HN
    i_r = HN[:, 0]
    i_theta = HN[:, 1]
    i_h = HN[:, 2]

    assert approx_equal(i_r, i_r_expected)
    assert approx_equal(i_theta, i_theta_expected)
    assert approx_equal(i_h, i_h_expected)


def test_initial_alignment():
    """Test that the Hill frame is aligned with the expected axes at time t = 0.
    1. At time t=0, the Hill frame (HN) should be aligned with the expected inertial vectors.
    2. The vectors i_r, i_theta, i_h should match the calculated values based on the initial orbital elements.
    """
    HN = Inertial2Hill(0)

    # initial angles
    angles = np.array([20, 30, 60]) * np.pi / 180
    rN, vN = orbitalElements2Inertial(r_LMO, angles)

    # expected Hill axes
    i_r_expected = rN / np.linalg.norm(rN)
    i_h_expected = np.cross(rN, vN)
    i_h_expected /= np.linalg.norm(i_h_expected)
    i_theta_expected = np.cross(i_h_expected, i_r_expected)

    assert approx_equal(HN[:, 0], i_r_expected)
    assert approx_equal(HN[:, 1], i_theta_expected)
    assert approx_equal(HN[:, 2], i_h_expected)


def test_continuity():
    """Test that the Direct Cosine Matrix (HN) changes continuously with time.
    1. Computes HN at two slightly different times (t and t + dt).
    2. The difference between HN at these two times should be small, confirming continuous motion.
    """
    t = 1000
    dt = 1e-3

    H1 = Inertial2Hill(t)
    H2 = Inertial2Hill(t + dt)

    diff = np.linalg.norm(H2 - H1)
    assert diff < 1e-3


def test_inverse_transform():
    """Test the inverse transformation between Hill and inertial frames.
    1. Given a random inertial vector, it should be possible to transform it to the Hill frame and then back to the inertial frame.
    2. The resulting vector should match the original inertial vector.
    """
    t = 300
    HN = Inertial2Hill(t)   # Hill → inertial
    NH = HN.T               # inertial → Hill

    v = np.array([1.2, -3.4, 2.1])   # random inertial vector

    v_hill = NH @ v
    v_back = HN @ v_hill

    assert approx_equal(v_back, v)


def test_h_fixed():
    """Test that the 'h' vector in the Hill frame remains constant over time.
    1. The third column of the Direct Cosine Matrix (HN) represents the 'h' vector.
    2. The 'h' vector should be fixed (i.e., it should not change with time).
    """
    t1 = 0
    t2 = 4000

    H1 = Inertial2Hill(t1)
    H2 = Inertial2Hill(t2)

    h1 = H1[:, 2]
    h2 = H2[:, 2]

    assert approx_equal(h2, h1)


def test_angular_rate():
    """Test the angular rate of the Hill frame.
    1. The angular rate of the Hill frame is determined by the time derivative of the Hill frame's i_theta vector.
    2. The angle between i_theta vectors at times t and t + dt should match the expected angular rate.
    """
    theta_dot = np.sqrt(mu / r_LMO**3)

    t = 500
    dt = 1.0  # 1-second step

    H1 = Inertial2Hill(t)
    H2 = Inertial2Hill(t + dt)

    i_theta1 = H1[:, 1]
    i_theta2 = H2[:, 1]

    # angle difference via dot product
    angle = np.arccos(np.clip(np.dot(i_theta1, i_theta2), -1, 1))
    expected = theta_dot * dt

    assert abs(angle - expected) < 1e-3


# Run all tests with pytest (no need to call explicitly, just run pytest in terminal)
# Example: pytest -v

