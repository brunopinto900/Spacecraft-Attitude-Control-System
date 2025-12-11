import numpy as np

def wrap_angle_rad_2pi(angle):
    """
    Wraps an angle in radians to [0, 2π).
    """
    return angle % (2 * np.pi)


# -------------------------
# Unit tests
# -------------------------
def test_wrap_angle_rad_2pi():
    tol = 1e-12  # tolerance for floating point comparisons

    # Test 0
    assert abs(wrap_angle_rad_2pi(0) - 0) < tol

    # Positive angle < 2π
    assert abs(wrap_angle_rad_2pi(np.pi/2) - np.pi/2) < tol

    # Negative angle
    assert abs(wrap_angle_rad_2pi(-np.pi/2) - (2*np.pi - np.pi/2)) < tol

    # Large positive angle
    assert abs(wrap_angle_rad_2pi(10*np.pi) - 0) < tol

    # Large negative angle
    assert abs(wrap_angle_rad_2pi(-7*np.pi) - np.pi) < tol

    # Value just below 2π
    x = 2*np.pi - 1e-12
    assert abs(wrap_angle_rad_2pi(x) - x) < tol

    # Exactly 2π
    assert abs(wrap_angle_rad_2pi(2*np.pi) - 0) < tol

    # Array input
    angles = np.array([0, -np.pi, 3*np.pi])
    expected = np.array([0, np.pi, np.pi])
    wrapped = wrap_angle_rad_2pi(angles)
    assert np.all(np.abs(wrapped - expected) < tol)

    print("All tests passed!")


# Run tests
if __name__ == "__main__":
    test_wrap_angle_rad_2pi()
