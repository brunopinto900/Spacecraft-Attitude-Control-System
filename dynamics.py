"""
Spacecraft dynamics and integration functions.

This module provides the equations of motion for spacecraft dynamics
and numerical integration routines (RK4).
"""

import numpy as np
from math_utils import tilde_matrix
from perturbations import gravity_acceleration_j2


def dynamics(x, d, u, omega_RW, t):
    """
    Spacecraft dynamics using Modified Rodrigues Parameters (MRP).
    
    Parameters
    ----------
    x : ndarray, shape (6,)
        State vector [sigma (3x1); omega_BN (3x1)] where
        sigma: MRP attitude
        omega_BN: angular velocity of B relative to N in B frame
    d : ndarray, shape (3,)
        Disturbance torque vector
    u : ndarray, shape (3,)
        Control torque vector
    omega_RW : ndarray, shape (3,)
        Reaction wheel angular velocities
    t : float
        Time (not used, included for ODE solver compatibility)
        
    Returns
    -------
    xdot : ndarray, shape (6,)
        Time derivative of state vector
    """
    from config import I_sat, I_b, I_RW1Bcg, I_RW2Bcg, I_RW3Bcg
    
    sigma = x[0:3]
    omega = x[3:6]
    
    sigma_n2 = np.dot(sigma.T, sigma)
   
    # MRP kinematics
    B = (1 - sigma_n2) * np.eye(3) + 2 * tilde_matrix(sigma) + 2 * np.outer(sigma, sigma.T)
    sigma_dot = 0.25 * B @ omega

    # Angular momentum
    H = I_sat @ omega + (I_RW1Bcg + I_RW2Bcg + I_RW3Bcg) @ omega_RW  # Total H with Reaction Wheels

    cross_term = tilde_matrix(omega) @ H  # shape (3,)
    omega_dot = np.linalg.inv(I_b) @ (u + d - cross_term)  # shape (3,)
    
    # Combine into xdot
    xdot = np.hstack((sigma_dot, omega_dot))
    return xdot


def RK4(xdot_func, x_t, d_t, u_t, omega_RW_t, t, dt):
    """
    Fourth-order Runge-Kutta integration step for attitude dynamics.
    
    Parameters
    ----------
    xdot_func : callable
        Function computing the time derivative of the state: xdot = xdot_func(x, d, u, omega_RW, t)
    x_t : ndarray
        Current state vector
    d_t : ndarray
        Torque disturbance at current time
    u_t : ndarray
        Control input at current time
    omega_RW_t : ndarray
        Reaction wheel angular velocities at current time
    t : float
        Current time
    dt : float
        Time step
        
    Returns
    -------
    x_next : ndarray
        State vector at time t + dt
    """
    k1 = xdot_func(x_t, d_t, u_t, omega_RW_t, t)
    k2 = xdot_func(x_t + k1 * dt / 2, d_t, u_t, omega_RW_t, t + dt / 2)
    k3 = xdot_func(x_t + k2 * dt / 2, d_t, u_t, omega_RW_t, t + dt / 2)
    k4 = xdot_func(x_t + k3 * dt, d_t, u_t, omega_RW_t, t + dt)

    x_next = x_t + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    return x_next


def rk4_gravity_step(acc_drag, r, v, dt, mu):
    """
    RK4 integration for position and velocity under gravity with J2 perturbation and drag.
    
    Parameters
    ----------
    acc_drag : array-like, shape (3,)
        Drag acceleration vector [km/s^2]
    r : array-like, shape (3,)
        Position vector [km]
    v : array-like, shape (3,)
        Velocity vector [km/s]
    dt : float
        Timestep [s]
    mu : float
        Gravitational parameter [km^3/s^2]
        
    Returns
    -------
    r_new : ndarray, shape (3,)
        Updated position
    v_new : ndarray, shape (3,)
        Updated velocity
    """
    def acceleration(pos):
        return gravity_acceleration_j2(pos) + acc_drag

    # k1
    k1_v = acceleration(r) * dt
    k1_r = v * dt

    # k2
    k2_v = acceleration(r + 0.5*k1_r) * dt
    k2_r = (v + 0.5*k1_v) * dt

    # k3
    k3_v = acceleration(r + 0.5*k2_r) * dt
    k3_r = (v + 0.5*k2_v) * dt

    # k4
    k4_v = acceleration(r + k3_r) * dt
    k4_r = (v + k3_v) * dt

    # Update
    r_new = r + (k1_r + 2*k2_r + 2*k3_r + k4_r)/6
    v_new = v + (k1_v + 2*k2_v + 2*k3_v + k4_v)/6

    return r_new, v_new
