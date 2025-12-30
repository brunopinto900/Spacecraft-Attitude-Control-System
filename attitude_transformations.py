"""
Attitude representation transformations for spacecraft.

This module provides functions for converting between different attitude
representations including Direction Cosine Matrices (DCM), Modified Rodrigues
Parameters (MRP), and quaternions.
"""

import numpy as np
from math_utils import tilde_matrix, normalize_mrp, normalize_dcm


def MRP2DCM(sigma):
    """
    Converts Modified Rodrigues Parameters (MRP) to DCM.
    
    Automatically handles shadow set if |σ| > 1.
    
    Parameters
    ----------
    sigma : array-like, shape (3,)
        MRP vector
        
    Returns
    -------
    ndarray, shape (3, 3)
        Rotation matrix (DCM)
    """
    sigma = np.asarray(sigma).reshape(3)
    sigma_squared = np.inner(sigma, sigma)
    DCM = np.eye(3) + (8*tilde_matrix(sigma)@tilde_matrix(sigma) - 4*(1 - sigma_squared)*tilde_matrix(sigma)) / (1 + sigma_squared)**2
    return DCM


def DCM2MRP(matrix):
    """
    Converts DCM to MRP (principal set, |σ| < 1).
    
    Automatically handles flips for large rotations.
    
    Parameters
    ----------
    matrix : ndarray, shape (3, 3)
        Rotation matrix (DCM)
        
    Returns
    -------
    ndarray, shape (3,)
        MRP vector
    """
    zeta = np.sqrt(np.trace(matrix) + 1)
    constant = 1 / (zeta**2 + 2 * zeta)
    s1 = constant * (matrix[1, 2] - matrix[2, 1])
    s2 = constant * (matrix[2, 0] - matrix[0, 2])
    s3 = constant * (matrix[0, 1] - matrix[1, 0])
    return np.array([s1, s2, s3])


def dcm2quat(C):
    """
    Convert rotation matrix (DCM) to quaternion [q0, q1, q2, q3] scalar-first.
    
    Parameters
    ----------
    C : ndarray, shape (3, 3)
        Direction Cosine Matrix
        
    Returns
    -------
    ndarray, shape (4,)
        Quaternion [scalar, vector] with scalar component first
    """
    C = np.asarray(C).reshape(3, 3)
    tr = np.trace(C)
    
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q0 = 0.25 * S
        q1 = (C[2, 1] - C[1, 2]) / S
        q2 = (C[0, 2] - C[2, 0]) / S
        q3 = (C[1, 0] - C[0, 1]) / S
    else:
        if (C[0, 0] > C[1, 1]) and (C[0, 0] > C[2, 2]):
            S = np.sqrt(1 + C[0, 0] - C[1, 1] - C[2, 2]) * 2
            q0 = (C[2, 1] - C[1, 2]) / S
            q1 = 0.25 * S
            q2 = (C[0, 1] + C[1, 0]) / S
            q3 = (C[0, 2] + C[2, 0]) / S
        elif C[1, 1] > C[2, 2]:
            S = np.sqrt(1 + C[1, 1] - C[0, 0] - C[2, 2]) * 2
            q0 = (C[0, 2] - C[2, 0]) / S
            q1 = (C[0, 1] + C[1, 0]) / S
            q2 = 0.25 * S
            q3 = (C[1, 2] + C[2, 1]) / S
        else:
            S = np.sqrt(1 + C[2, 2] - C[0, 0] - C[1, 1]) * 2
            q0 = (C[1, 0] - C[0, 1]) / S
            q1 = (C[0, 2] + C[2, 0]) / S
            q2 = (C[1, 2] + C[2, 1]) / S
            q3 = 0.25 * S
    
    q = np.array([q0, q1, q2, q3])
    # Enforce scalar positive
    if q[0] < 0:
        q = -q
    return q


def DCM2MRP_alt(C):
    """
    Convert rotation matrix (DCM) to MRP, normalized deterministically.
    
    Alternative implementation using quaternion intermediate representation.
    
    Parameters
    ----------
    C : ndarray, shape (3, 3)
        Direction Cosine Matrix
        
    Returns
    -------
    ndarray, shape (3,)
        Normalized MRP vector
    """
    b = dcm2quat(C)  # quaternion scalar-first
    q = np.zeros(3)
    q[0] = b[1] / (1 + b[0])
    q[1] = b[2] / (1 + b[0])
    q[2] = b[3] / (1 + b[0])
    return normalize_mrp(q)


def Rscrew(n):
    """
    Rotation matrix that aligns wheel's spin axis with direction n.
    
    Equivalent to MATLAB Rscrew(n): builds an orthonormal frame where
    the first column is aligned with n.
    
    Parameters
    ----------
    n : array-like, shape (3,)
        Direction vector (will be normalized)
        
    Returns
    -------
    ndarray, shape (3, 3)
        Rotation matrix with columns as RW frame axes in body frame
    """
    n = np.asarray(n).reshape(3)
    n = n / np.linalg.norm(n)

    # Choose an arbitrary vector that is not parallel to n
    if abs(n[0]) < 0.9:
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])

    # Build orthonormal frame: n, y, z
    y = np.cross(n, v)
    y /= np.linalg.norm(y)
    z = np.cross(n, y)
    z /= np.linalg.norm(z)

    # Columns are the RW frame axes expressed in body frame
    return np.column_stack((n, y, z))
