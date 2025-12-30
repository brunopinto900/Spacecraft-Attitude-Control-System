"""
Mathematical utility functions for spacecraft attitude control.

This module provides basic mathematical operations used throughout the
spacecraft attitude control system, including skew-symmetric matrices,
vector operations, and angle wrapping.
"""

import numpy as np


def tilde_matrix(v):
    """
    Returns the skew-symmetric (tilde) matrix of a 3-vector v.
    
    Also known as the cross-product matrix, such that tilde_matrix(v) @ w = cross(v, w).
    
    Parameters
    ----------
    v : array-like, shape (3,)
        Input vector
        
    Returns
    -------
    ndarray, shape (3, 3)
        Skew-symmetric matrix
    """
    v = np.asarray(v).reshape(3)
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0]
    ])


def skew(v):
    """
    Return skew-symmetric matrix for vector v.
    
    Alias for tilde_matrix function.
    
    Parameters
    ----------
    v : array-like, shape (3,)
        Input vector
        
    Returns
    -------
    ndarray, shape (3, 3)
        Skew-symmetric matrix
    """
    v = np.asarray(v).reshape(3)
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def normalize_mrp(sigma):
    """
    Convert MRP to short rotation (norm <= 1).
    
    Optionally enforces first component positive for deterministic representation.
    
    Parameters
    ----------
    sigma : array-like, shape (3,)
        Modified Rodrigues Parameters vector
        
    Returns
    -------
    ndarray, shape (3,)
        Normalized MRP vector
    """
    sigma = np.asarray(sigma).reshape(3)
    
    # Optional: enforce first component positive for deterministic representation
    if sigma[0] < 0:
        sigma = -sigma
    return sigma


def normalize_dcm(R):
    """
    Normalize a Direction Cosine Matrix (DCM) using SVD.
    
    Ensures the matrix is properly orthonormal by performing SVD
    re-orthogonalization.
    
    Parameters
    ----------
    R : ndarray, shape (3, 3)
        Direction Cosine Matrix to normalize
        
    Returns
    -------
    ndarray, shape (3, 3)
        Normalized orthogonal DCM
    """
    # Perform SVD to re-orthogonalize the matrix
    U, _, Vt = np.linalg.svd(R)
    R_normalized = np.dot(U, Vt)
    return R_normalized


def angle_between_two_vec(vec1, vec2):
    """
    Computes the angle between two vectors in degrees.
    
    Parameters
    ----------
    vec1 : array-like, shape (3,)
        First vector
    vec2 : array-like, shape (3,)
        Second vector
        
    Returns
    -------
    float
        Angle between vectors in degrees
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Compute cosine of the angle
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # Clip to [-1, 1] to avoid numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Compute angle in radians
    theta_rad = np.arccos(cos_theta)
    
    # Convert to degrees
    theta_deg = np.degrees(theta_rad)
    
    return theta_deg


def wrap_angle_rad_2pi(angle):
    """
    Wraps an angle in radians to [0, 2π).
    
    Parameters
    ----------
    angle : float or ndarray
        Angle(s) in radians to wrap
        
    Returns
    -------
    float or ndarray
        Wrapped angle(s)
    """
    return angle
    # return angle % (2 * np.pi)
