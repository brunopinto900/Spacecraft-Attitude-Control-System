"""
Environmental perturbation models for spacecraft.

This module provides functions for calculating disturbance forces and torques
including atmospheric drag, gravity gradient, magnetic torques, and J2 perturbation.
"""

import numpy as np
from attitude_transformations import MRP2DCM


def earth_thermosphere_density(alt_km):
    """
    Returns a realistic Earth atmospheric density (kg/m^3) for LEO altitudes.
    
    Approximate piecewise exponential fit for 150-600 km based on NRLMSISE-00 typical values.
    
    Parameters
    ----------
    alt_km : float
        Altitude in kilometers
        
    Returns
    -------
    float
        Atmospheric density in kg/m^3
    """
    h = alt_km
    if h < 150:
        rho = 3.614e-9
    elif h < 200:
        rho = 2.789e-10
    elif h < 250:
        rho = 7.248e-11
    elif h < 300:
        rho = 2.418e-11
    elif h < 350:
        rho = 9.518e-12
    elif h < 400:
        rho = 3.725e-12
    elif h < 450:
        rho = 1.585e-12
    elif h < 500:
        rho = 6.967e-13
    elif h < 550:
        rho = 3.056e-13
    elif h < 600:
        rho = 1.350e-13
    else:
        rho = 1e-13  # Above 600 km, very tenuous
    return rho


def drag_force(sigma_BN, r_i, v_i):
    """
    Calculate atmospheric drag force using MRPs.
    
    Parameters
    ----------
    sigma_BN : array-like, shape (3,)
        MRP attitude parameters (body wrt inertial)
    r_i : array-like, shape (3,)
        Position in inertial frame [km]
    v_i : array-like, shape (3,)
        Velocity in inertial frame [km/s]
        
    Returns
    -------
    ndarray, shape (3,)
        Drag force in body frame [N]
    """
    from config import mu, r_LMO, Lx, Ly, Lz, h, C_d
    
    # Planet rotation rate
    theta_dot = np.sqrt(mu/r_LMO**3)  # rad/s
    omega = np.array([0, 0, theta_dot])

    # Convert MRP to DCM
    C_BN = MRP2DCM(sigma_BN)

    # Compute atmospheric-relative velocity in body frame
    r_i = np.array(r_i) * 1000  # Convert to meters
    v_i = np.array(v_i) * 1000  # Convert to m/s

    v_atm = np.cross(omega, r_i)
    v_rel_i = v_i - v_atm
    v_rel_b = C_BN @ v_rel_i

    v_mag = np.linalg.norm(v_rel_b)
    if v_mag == 0:
        return np.zeros(3)

    v_hat_b = v_rel_b / v_mag

    # Dynamic center of pressure (projected area)
    normals = {
        "+x": np.array([ 1, 0, 0]),
        "-x": np.array([-1, 0, 0]),
        "+y": np.array([ 0, 1, 0]),
        "-y": np.array([ 0,-1, 0]),
        "+z": np.array([ 0, 0, 1]),
        "-z": np.array([ 0, 0,-1])
    }

    face_centers = {
        "+x": np.array([ Lx/2, 0,      0]),
        "-x": np.array([-Lx/2, 0,      0]),
        "+y": np.array([ 0,      Ly/2, 0]),
        "-y": np.array([ 0,     -Ly/2, 0]),
        "+z": np.array([ 0,      0,     Lz/2]),
        "-z": np.array([ 0,      0,    -Lz/2])
    }

    areas = {
        "+x": Ly*Lz, "-x": Ly*Lz,
        "+y": Lx*Lz, "-y": Lx*Lz,
        "+z": Lx*Ly, "-z": Lx*Ly
    }

    weights, centers = [], []

    for key in normals:
        proj = np.dot(v_hat_b, normals[key])
        if proj < 0:  # wind-facing
            A_proj = areas[key] * abs(proj)
            weights.append(A_proj)
            centers.append(face_centers[key])

    if len(weights) == 0:
        r_cp = np.zeros(3)
    else:
        weights = np.array(weights)
        centers = np.array(centers)
        r_cp = (weights[:, None] * centers).sum(axis=0) / weights.sum()

    # Drag force model
    A_ref = max(Lx*Ly, Ly*Lz, Lx*Lz)
    rho = earth_thermosphere_density(h)
    F_drag_b = -0.5 * rho * v_mag**2 * C_d * A_ref * v_hat_b
    
    return F_drag_b


def drag_torque(sigma_BN, r_i, v_i):
    """
    Calculate atmospheric drag torque using MRPs.
    
    Parameters
    ----------
    sigma_BN : array-like, shape (3,)
        MRP attitude parameters (body wrt inertial)
    r_i : array-like, shape (3,)
        Position in inertial frame [km]
    v_i : array-like, shape (3,)
        Velocity in inertial frame [km/s]
        
    Returns
    -------
    ndarray, shape (3,)
        Drag torque in body frame [N·m]
    """
    from config import mu, r_LMO, Lx, Ly, Lz, h, C_d
    
    # Planet rotation rate
    theta_dot = np.sqrt(mu/r_LMO**3)  # rad/s
    omega = np.array([0, 0, theta_dot])

    # Convert MRP to DCM
    C_BN = MRP2DCM(sigma_BN)

    # Compute atmospheric-relative velocity in body frame
    r_i = np.array(r_i) * 1000  # Convert to meters
    v_i = np.array(v_i) * 1000  # Convert to m/s

    v_atm = np.cross(omega, r_i)
    v_rel_i = v_i - v_atm
    v_rel_b = C_BN @ v_rel_i

    v_mag = np.linalg.norm(v_rel_b)
    if v_mag == 0:
        return np.zeros(3)

    v_hat_b = v_rel_b / v_mag

    # Dynamic center of pressure (projected area)
    normals = {
        "+x": np.array([ 1, 0, 0]),
        "-x": np.array([-1, 0, 0]),
        "+y": np.array([ 0, 1, 0]),
        "-y": np.array([ 0,-1, 0]),
        "+z": np.array([ 0, 0, 1]),
        "-z": np.array([ 0, 0,-1])
    }

    face_centers = {
        "+x": np.array([ Lx/2, 0,      0]),
        "-x": np.array([-Lx/2, 0,      0]),
        "+y": np.array([ 0,      Ly/2, 0]),
        "-y": np.array([ 0,     -Ly/2, 0]),
        "+z": np.array([ 0,      0,     Lz/2]),
        "-z": np.array([ 0,      0,    -Lz/2])
    }

    areas = {
        "+x": Ly*Lz, "-x": Ly*Lz,
        "+y": Lx*Lz, "-y": Lx*Lz,
        "+z": Lx*Ly, "-z": Lx*Ly
    }

    weights, centers = [], []

    for key in normals:
        proj = np.dot(v_hat_b, normals[key])
        if proj < 0:  # wind-facing
            A_proj = areas[key] * abs(proj)
            weights.append(A_proj)
            centers.append(face_centers[key])

    if len(weights) == 0:
        r_cp = np.zeros(3)
    else:
        weights = np.array(weights)
        centers = np.array(centers)
        r_cp = (weights[:, None] * centers).sum(axis=0) / weights.sum()

    # Drag force model
    A_ref = max(Lx*Ly, Ly*Lz, Lx*Lz)
    rho = earth_thermosphere_density(h)
    F_drag_b = -0.5 * rho * v_mag**2 * C_d * A_ref * v_hat_b

    # Drag torque in body frame
    tau = np.max(r_cp) * F_drag_b
    return tau


def gravity_gradient_torque(rN, sigma_BN):
    """
    Compute gravity-gradient torque in the body frame.
    
    Parameters
    ----------
    rN : array-like, shape (3,)
        Satellite position in inertial frame [km]
    sigma_BN : array-like, shape (3,)
        MRP vector describing body -> inertial rotation
        
    Returns
    -------
    ndarray, shape (3,)
        Gravity gradient torque in body frame [N·m]
    """
    from config import mu, I_b
    
    # DCM: body -> inertial
    C_BN = MRP2DCM(sigma_BN)
    
    # Position in body frame
    r_b = C_BN @ rN
    r_norm = np.linalg.norm(r_b)
    r_hat = r_b / r_norm
    
    # Gravity gradient torque
    T_gg = 3 * mu / r_norm**3 * np.cross(r_hat, I_b @ r_hat)
    
    return T_gg


def B_eci_dipole(rN):
    """
    Compute Earth's magnetic field in ECI using a simple dipole model.
    
    Parameters
    ----------
    rN : array-like, shape (3,)
        Position vector in ECI [km]
        
    Returns
    -------
    ndarray, shape (3,)
        Magnetic field in ECI [T]
    """
    r = np.array(rN)
    r_norm = np.linalg.norm(r)
    r_hat = r / r_norm
    z_hat = np.array([0, 0, 1])
    
    # Earth's magnetic moment magnitude (approx)
    B0 = 3.12e-5  # Tesla at reference radius (~Earth surface)
    R_E = 6371.0  # km
    # Dipole scaling
    factor = B0 * (R_E / r_norm)**3
    
    B_eci = factor * (3 * np.dot(r_hat, z_hat) * r_hat - z_hat)
    return B_eci


def magnetic_torque_igrf(rN, sigma_BN):
    """
    Compute magnetic torque in body frame without recomputing IGRF.
    
    Parameters
    ----------
    rN : array-like, shape (3,)
        Position vector in inertial frame [km]
    sigma_BN : array-like, shape (3,)
        MRP attitude vector (body -> inertial)
        
    Returns
    -------
    ndarray, shape (3,)
        Magnetic torque in body frame [N·m]
    """
    # Satellite magnetic dipole in body frame [A·m^2]
    m_b = np.array([2.64/1000, 2.64/1000, 2.64/1000])

    B_eci = B_eci_dipole(rN)

    # Convert B field from inertial to body frame
    C_BN = MRP2DCM(sigma_BN)
    B_b = C_BN @ B_eci
    
    # Ensure flat vectors
    m_b = np.array(m_b).reshape(3)
    B_b = np.array(B_b).reshape(3)
    
    # Torque
    T_mag_b = np.cross(m_b, B_b)
    return T_mag_b


def gravity_acceleration_j2(r):
    """
    Compute gravitational acceleration with J2 perturbation.
    
    Parameters
    ----------
    r : array-like, shape (3,)
        Position vector in inertial frame [km]
        
    Returns
    -------
    ndarray, shape (3,)
        Acceleration vector [km/s^2]
    """
    from config import mu, R_planet
    
    J2 = 1.08263e-3

    x, y, z = r
    r_norm = np.linalg.norm(r)
    
    # Central gravity
    a_central = -mu * r / r_norm**3
    
    # J2 perturbation
    z2 = z**2
    r2 = r_norm**2
    factor = 1.5 * J2 * mu * R_planet**2 / r_norm**5
    
    a_j2 = factor * np.array([
        x * (5*z2/r2 - 1),
        y * (5*z2/r2 - 1),
        z * (5*z2/r2 - 3)
    ])
    
    return a_central + a_j2


def calculateTorqueDisturbances(sigma_BN, r_i, v_i):
    """
    Calculate total disturbance torques acting on the spacecraft.
    
    Parameters
    ----------
    sigma_BN : array-like, shape (3,)
        MRP attitude parameters
    r_i : array-like, shape (3,)
        Position in inertial frame [km]
    v_i : array-like, shape (3,)
        Velocity in inertial frame [km/s]
        
    Returns
    -------
    ndarray, shape (3,)
        Total disturbance torque [N·m]
    """
    # Atmospheric (Drag) Torque
    aero_drag_torque = drag_torque(sigma_BN, r_i, v_i)

    # Gravity Gradient
    gravity_grad_torque = gravity_gradient_torque(r_i, sigma_BN)

    # Magnetic Dipole
    mag_torque = magnetic_torque_igrf(r_i, sigma_BN)

    disturbance = aero_drag_torque + gravity_grad_torque + mag_torque

    return disturbance


def calculateForceDisturbances(sigma_BN, r_i, v_i):
    """
    Calculate total disturbance forces acting on the spacecraft.
    
    Parameters
    ----------
    sigma_BN : array-like, shape (3,)
        MRP attitude parameters
    r_i : array-like, shape (3,)
        Position in inertial frame [km]
    v_i : array-like, shape (3,)
        Velocity in inertial frame [km/s]
        
    Returns
    -------
    ndarray, shape (3,)
        Total disturbance force in inertial frame [N]
    """
    # Atmospheric (Drag) Force
    F_drag_body = drag_force(sigma_BN, r_i, v_i)
    C_BN = MRP2DCM(sigma_BN)  # body -> inertial
    F_drag_inertial = C_BN.T @ F_drag_body
    return F_drag_inertial
