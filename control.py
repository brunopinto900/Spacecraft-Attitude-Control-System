"""
Control algorithms for spacecraft attitude control.

This module provides functions for calculating control inputs, error computation,
and reaction wheel control.
"""

import numpy as np
from attitude_transformations import MRP2DCM, DCM2MRP
from math_utils import tilde_matrix


def calculateError(measured_body_attitude_mrp,
                   measured_angular_velocity,
                   reference_attitude_DCM,
                   reference_angular_velocity):
    """
    Compute attitude and angular velocity tracking errors between the body frame and a reference frame.
    
    Parameters
    ----------
    measured_body_attitude_mrp : array-like, shape (3,)
        MRP attitude of body frame B relative to inertial frame N
    measured_angular_velocity : array-like, shape (3,)
        Angular velocity of B relative to N, expressed in B frame
    reference_attitude_DCM : ndarray, shape (3, 3)
        DCM of reference frame R relative to N
    reference_angular_velocity : array-like, shape (3,)
        Angular velocity of R relative to N, expressed in N frame
        
    Returns
    -------
    beta_BR : ndarray, shape (3,)
        Attitude error of B relative to R in MRP form
    wB_BR : ndarray, shape (3,)
        Angular velocity tracking error of B relative to R, expressed in B frame
    """
    # Convert body MRP to DCM
    BN = MRP2DCM(measured_body_attitude_mrp)

    # Relative rotation from B to R
    RB = BN @ reference_attitude_DCM.T
    beta_BR = DCM2MRP(RB)  # convert relative rotation to MRP

    # Angular velocity tracking error
    wB_BR = measured_angular_velocity - BN @ reference_angular_velocity

    return beta_BR, wB_BR


def omega_dot(omega, I, u):
    """
    Compute the time derivative of angular velocity for a rigid body.
    
    Automatically handles 1D or 2D column array inputs.
    
    Parameters
    ----------
    omega : array-like, shape (3,) or (3, 1)
        Angular velocity in body frame (rad/s)
    I : ndarray, shape (3, 3)
        Inertia matrix of the body
    u : array-like, shape (3,) or (3, 1)
        Control torque applied in body frame
        
    Returns
    -------
    ndarray, shape (3,)
        Time derivative of angular velocity
    """
    # Ensure omega and u are 1D arrays
    omega = np.asarray(omega).reshape(3)
    u = np.asarray(u).reshape(3)

    # Angular momentum
    H = I @ omega  # shape (3,)

    # Gyroscopic / cross product term
    cross_term = tilde_matrix(omega) @ H  # shape (3,)

    # Angular velocity derivative
    w_dot = np.linalg.inv(I) @ (u - cross_term)  # shape (3,)

    return w_dot


def reaction_wheel_torque(omega_RW, rwalphas):
    """
    Compute reaction wheel angular accelerations and resulting body torque.
    
    Parameters
    ----------
    omega_RW : array-like, shape (3,)
        RW speeds [rad/s]
    rwalphas : array-like, shape (3,)
        Commanded RW accelerations [rad/s²]
        
    Returns
    -------
    LMN_RWs : ndarray, shape (3,)
        Reaction wheel torque on spacecraft body [N·m]
    w123dot : ndarray, shape (3,)
        Resulting angular accelerations [rad/s²]
    """
    from config import maxSpeed, maxTorque, I_RW, Ir1B, Ir2B, Ir3B
    
    # Maximum angular acceleration
    maxAlpha = maxTorque
    w123 = np.asarray(omega_RW).reshape(3)
    rwalphas = np.asarray(rwalphas).reshape(3)

    w123dot = np.zeros(3)

    for i in range(3):
        # Speed saturation
        if abs(w123[i]) > maxSpeed:
            w123dot[i] = 0.0
        else:
            # Acceleration saturation
            if abs(rwalphas[i]) > maxAlpha:
                rwalphas[i] = np.sign(rwalphas[i]) * maxAlpha
            
            w123dot[i] = rwalphas[i]

    w123dot = rwalphas
    LMN_RWs = (Ir1B + Ir2B + Ir3B) @ w123dot

    return LMN_RWs, w123dot


def calculateControl(omega_RW, attitudeError, angularVelError):
    """
    Calculate control torque using PD control law.
    
    Parameters
    ----------
    omega_RW : array-like, shape (3,)
        Reaction wheel angular velocities [rad/s]
    attitudeError : array-like, shape (3,)
        Attitude error in MRP form
    angularVelError : array-like, shape (3,)
        Angular velocity error [rad/s]
        
    Returns
    -------
    u : ndarray, shape (3,)
        Control torque [N·m]
    accel_RW : ndarray, shape (3,)
        Reaction wheel accelerations [rad/s²]
    """
    from config import I_b, Ir1B, Ir2B, Ir3B
    
    P = np.max(I_b * (2/120))
    K = (P**2) / I_b[1, 1]

    # Reaction Wheel
    desiredTorque = -K * attitudeError - P * angularVelError
    desired_accel_RW = np.linalg.inv((Ir1B + Ir2B + Ir3B)) @ desiredTorque
    u, accel_RW = reaction_wheel_torque(omega_RW, desired_accel_RW)

    return u, accel_RW
