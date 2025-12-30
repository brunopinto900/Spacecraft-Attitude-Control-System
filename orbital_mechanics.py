"""
Orbital mechanics calculations for spacecraft.

This module provides functions for orbital element conversions, coordinate
frame transformations, and orbital propagation.
"""

import numpy as np
from math_utils import normalize_dcm, wrap_angle_rad_2pi


def orbitalElements2Inertial(r, angles):
    """
    Convert orbital elements to inertial position and velocity.
    
    Parameters
    ----------
    r : float
        Orbital radius [km]
    angles : array-like, shape (3,)
        [RAAN, inclination, true_anomaly] in radians
        
    Returns
    -------
    rN : ndarray, shape (3,)
        Position vector in inertial frame [km]
    vN : ndarray, shape (3,)
        Velocity vector in inertial frame [km/s]
    """
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
    
    # Import mu from config at function call to avoid circular import
    from config import mu
    n = np.sqrt(mu / r**3)
    v_perif = np.array([0, r*n, 0])

    rN = perif_to_ECI @ r_perif
    vN = perif_to_ECI @ v_perif

    return rN, vN


def Inertial2Hill(t):
    """
    Computes the Direct Cosine Matrix from the Hill Frame (LMO) to the inertial frame (N).
    
    Parameters
    ----------
    t : float
        Time from t_0, in seconds
        
    Returns
    -------
    ndarray, shape (3, 3)
        Direct Cosine Matrix HN (Hill to Inertial)
    """
    # Import parameters from config
    from config import mu, r_LMO, RAAN, inclination, mean_anomaly
    
    # Initial conditions for LMO
    theta_dot_LMO = np.sqrt(mu/r_LMO**3)  # rad/sec
    angles_LMO = np.array([RAAN, inclination, mean_anomaly]) * np.pi/180  # radians
    angles_LMO += np.array([0, 0, theta_dot_LMO*t])  # Ω, i, θ(t), in radians
    angles_LMO = wrap_angle_rad_2pi(angles_LMO)
    
    # Computing inertial vectors
    r_inertial_LMO, r_dot_inertial_LMO = orbitalElements2Inertial(r_LMO, angles_LMO)
    
    # Computing Hill Frame versors
    i_r = r_inertial_LMO / np.linalg.norm(r_inertial_LMO)
    i_h = np.cross(r_inertial_LMO, r_dot_inertial_LMO) / np.linalg.norm(
        np.cross(r_inertial_LMO, r_dot_inertial_LMO))
    i_theta = np.cross(i_h, i_r)
    NH = np.array([i_r, i_theta, i_h])
    HN = NH.T
    return normalize_dcm(HN)
