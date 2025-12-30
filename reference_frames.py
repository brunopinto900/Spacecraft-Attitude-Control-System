"""
Reference frame transformations for spacecraft attitude control.

This module provides coordinate transformations between various reference frames
including inertial, sun-pointing, nadir-pointing, and communications frames, along
with their associated angular velocities.
"""

import numpy as np
from math_utils import wrap_angle_rad_2pi, normalize_dcm
from orbital_mechanics import orbitalElements2Inertial, Inertial2Hill


def init2Inertial():
    """
    Returns identity transformation for initialization mode.
    
    Returns
    -------
    ndarray, shape (3, 3)
        Identity matrix (no rotation)
    """
    R = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    return R


def Sun2Inertial():
    """
    Returns the transformation matrix from Sun-pointing reference frame (Rs) to the inertial frame (N).
    
    The Rs-frame is defined as:
        Rs-frame = [-n1, n2 x (-n1), n2]
    where the Sun-pointing frame is stationary relative to N.
    
    Returns
    -------
    ndarray, shape (3, 3)
        Rotation matrix from N-frame to Rs-frame
    """
    R = np.array([
        [-1, 0, 0],
        [ 0, 0, 1],
        [ 0, 1, 0]
    ])
    return R


def computeAngularVelocitySun():
    """
    Computes the angular velocity with respect to inertial frame written in Inertial Reference Frame.
    
    For the sun-pointing mode, the reference frame is stationary.
    
    Returns
    -------
    ndarray, shape (3,)
        Angular velocity w_RsN at time t (zero for stationary frame)
    """
    return np.array([0, 0, 0])


def Nadir2Inertial(t):
    """
    Computes the Direct Cosine Matrix from the Nadir Reference Frame (Rn) to the inertial frame (N).
    
    Parameters
    ----------
    t : float
        Time from t_0, in seconds
        
    Returns
    -------
    ndarray, shape (3, 3)
        Direct Cosine Matrix RnN
    """
    # Defining the versors from Rn in the Hill Frame (LMO) to construct the DCM [HRn]
    r_1 = np.array([-1, 0, 0])
    r_2 = np.array([0, 1, 0])
    r_3 = np.array([0, 0, -1])
    DCM_HRn = np.array([r_1, r_2, r_3])
    
    # [RnN] = [RnH][HN]
    RnN = DCM_HRn @ Inertial2Hill(t).T
    return normalize_dcm(RnN)


def computeAngularVelocityNadir(t):
    """
    Computes the angular velocity with respect to inertial frame written in Inertial Reference Frame.
    
    Parameters
    ----------
    t : float
        Time from t_0, in seconds
        
    Returns
    -------
    ndarray, shape (3,)
        Angular velocity w_RnN at time t
    """
    from config import mu, r_LMO
    
    # Initial conditions for LMO
    theta_dot_LMO = np.sqrt(mu/r_LMO**3)  # rad/sec
    
    # w normally would be, in the H frame:
    w_HN_H = np.array([0, 0, theta_dot_LMO])
    
    # Now we just compute that in the N frame
    w_HN_N = Inertial2Hill(t).T @ w_HN_H
   
    return w_HN_N


def Comms2Inertial(t):
    """
    Computes the Direct Cosine Matrix from the GMO-Pointing Reference Frame (Rc) to the inertial frame (N).
    
    Parameters
    ----------
    t : float
        Time from t_0, in seconds
        
    Returns
    -------
    ndarray, shape (3, 3)
        Direct Cosine Matrix RcN
    """
    from config import mu, r_LMO, r_GMO, RAAN, inclination, mean_anomaly
    
    # LMO
    theta_dot_LMO = np.sqrt(mu/r_LMO**3)  # rad/sec
    angles_LMO = np.array([RAAN, inclination, mean_anomaly]) * np.pi/180  # radians
    angles_LMO[2] = angles_LMO[2] + theta_dot_LMO*t  # Ω, i, θ(t), in radians
    angles_LMO = wrap_angle_rad_2pi(angles_LMO)
    
    # GMO
    theta_dot_GMO = np.sqrt(mu/r_GMO**3)  # rad/s
    angles_GMO = np.array([0, 0, 250]) * np.pi/180  # Ω, i, θ(t_0), in radians
    angles_GMO += np.array([0, 0, theta_dot_GMO*t])  # Ω, i, θ(t), in radians
    angles_GMO = wrap_angle_rad_2pi(angles_GMO)
    
    # Building delta_r
    r_GMO_inertial, v_GMO_inertial = orbitalElements2Inertial(r_GMO, angles_GMO)
    r_LMO_inertial, v_LMO_inertial = orbitalElements2Inertial(r_LMO, angles_LMO)
    delta_r = r_GMO_inertial - r_LMO_inertial
    
    # Rc frame
    n_3 = np.array([0, 0, 1])
    rc_1 = -delta_r / np.linalg.norm(delta_r)
    rc_2 = np.cross(delta_r, n_3) / np.linalg.norm(np.cross(delta_r, n_3))
    rc_3 = np.cross(rc_1, rc_2)
    
    DCM_RcN = np.array([rc_1, rc_2, rc_3]).T
    return DCM_RcN


def computeAngularVelocityComms(t):
    """
    Computes the angular velocity from the GMO-Pointing Reference Frame (Rc) to the inertial frame (N) written in N.
    
    Uses numerical differentiation to compute d/dt(RcN).
    
    Parameters
    ----------
    t : float
        Time from t_0, in seconds
        
    Returns
    -------
    ndarray, shape (3,)
        Angular velocity w_RcN at time t
    """
    from config import dt
    
    # d/dt(RcN) = (Rc(t+dt) - Rc(t))/dt
    RcN_dot = (Comms2Inertial(t+dt) - Comms2Inertial(t)) / dt
    w_tilde = -Comms2Inertial(t).T @ RcN_dot
    w = np.array([-w_tilde[1][2], w_tilde[0][2], -w_tilde[0][1]])
    return w
