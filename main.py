import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from animate3D import animate
from config import * # Import the constants from config.py



# Simulation Time Parameters
orbitRadius = R_planet + h
orbitPeriod = 2 * np.pi * np.sqrt(orbitRadius**3 / mu)  # seconds #400 #
numberOfOrbits = 1
simTime = numberOfOrbits * orbitPeriod
print("SIM TIME (1 ORBIT)")
print(simTime)
dt = 1.0
nPoints = simTime #6500
time_plot = np.linspace(0,nPoints,int(simTime/dt)+2)

def wrap_angle_rad_2pi(angle):
    """
    Wraps an angle in radians to [0, 2π).
    """
    return angle
    #return angle % (2 * np.pi)

# Example function to normalize a DCM
def normalize_dcm(R):
    # Perform SVD to re-orthogonalize the matrix
    U, _, Vt = np.linalg.svd(R)
    R_normalized = np.dot(U, Vt)
    return R_normalized

# -------------------------
# Normalize MRP (shadow set)
# -------------------------
def normalize_mrp(sigma):
    """
    Convert MRP to short rotation (norm <= 1).
    Also optionally enforce first component positive.
    """
    #if np.linalg.norm(sigma) > 1:
    #    sigma = -sigma / np.dot(sigma, sigma)
    #Optional: enforce first component positive
    if sigma[0] < 0:
        sigma = -sigma
    return sigma


def MRP2DCM(sigma):
    """
    Converts Modified Rodrigues Parameters (MRP) to DCM.
    Automatically handles shadow set if |σ| > 1.

    Arguments:
        sigma : np.array, shape (3,) -- MRP vector

    Returns:
        R : np.array, shape (3,3) -- rotation matrix (DCM)
    """
    sigma_squared = np.inner(sigma, sigma)
    DCM = np.eye(3) + (8*tilde_matrix(sigma)@tilde_matrix(sigma) - 4*(1 - sigma_squared)*tilde_matrix(sigma) )/ (1 + sigma_squared)**2
    return DCM

# -------------------------
# MRP -> DCM
# -------------------------
def MRP2DCM_(sigma):
    """
    Convert Modified Rodrigues Parameters (MRPs) to Direction Cosine Matrix (DCM)
    """
    sigma = normalize_mrp(sigma)
    s2 = sigma @ sigma
    S = tilde_matrix(sigma)
    C = np.eye(3) + (8 * (S @ S) - 4 * (1 - s2) * S) / (1 + s2)**2
    return C


# -------------------------
# DCM -> Quaternion
# -------------------------
def dcm2quat(C):
    """
    Convert rotation matrix (DCM) to quaternion [q0, q1, q2, q3] scalar-first
    """
    C = np.asarray(C).reshape(3,3)
    tr = np.trace(C)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q0 = 0.25 * S
        q1 = (C[2,1] - C[1,2]) / S
        q2 = (C[0,2] - C[2,0]) / S
        q3 = (C[1,0] - C[0,1]) / S
    else:
        if (C[0,0] > C[1,1]) and (C[0,0] > C[2,2]):
            S = np.sqrt(1 + C[0,0] - C[1,1] - C[2,2]) * 2
            q0 = (C[2,1] - C[1,2]) / S
            q1 = 0.25 * S
            q2 = (C[0,1] + C[1,0]) / S
            q3 = (C[0,2] + C[2,0]) / S
        elif C[1,1] > C[2,2]:
            S = np.sqrt(1 + C[1,1] - C[0,0] - C[2,2]) * 2
            q0 = (C[0,2] - C[2,0]) / S
            q1 = (C[0,1] + C[1,0]) / S
            q2 = 0.25 * S
            q3 = (C[1,2] + C[2,1]) / S
        else:
            S = np.sqrt(1 + C[2,2] - C[0,0] - C[1,1]) * 2
            q0 = (C[1,0] - C[0,1]) / S
            q1 = (C[0,2] + C[2,0]) / S
            q2 = (C[1,2] + C[2,1]) / S
            q3 = 0.25 * S
    q = np.array([q0, q1, q2, q3])
    # enforce scalar positive
    if q[0] < 0:
        q = -q
    return q


def DCM2MRP(matrix):
    """
    Converts DCM to MRP (principal set, |σ| < 1).
    Automatically handles flips for large rotations.

    Arguments:
        R : np.array, shape (3,3) -- rotation matrix

    Returns:
        sigma : np.array, shape (3,) -- MRP vector
    """
    zeta = np.sqrt(np.trace(matrix) + 1)
    constant = 1 / (zeta**2 + 2 * zeta)
    s1 = constant * (matrix[1, 2] - matrix[2, 1])
    s2 = constant * (matrix[2, 0] - matrix[0, 2])
    s3 = constant * (matrix[0, 1] - matrix[1, 0])
    return np.array([s1, s2, s3])

# -------------------------
# DCM -> MRP
# -------------------------
def DCM2MRP_(C):
    """
    Convert rotation matrix (DCM) to MRP, normalized deterministically
    """
    b = dcm2quat(C)  # quaternion scalar-first
    q = np.zeros(3)
    q[0] = b[1] / (1 + b[0])
    q[1] = b[2] / (1 + b[0])
    q[2] = b[3] / (1 + b[0])
    return normalize_mrp(q)


def calculateError(measured_body_attitude_mrp,
                  measured_angular_velocity,
                  reference_attitude_DCM,
                  reference_angular_velocity):
    """
    Compute attitude and angular velocity tracking errors between the body frame and a reference frame.

    Parameters
    ----------
    measured_body_attitude_mrp : (3,) array
        MRP attitude of body frame B relative to inertial frame N
    measured_angular_velocity : (3,) array
        Angular velocity of B relative to N, expressed in B frame
    reference_attitude_DCM : (3,3) array
        DCM of reference frame R relative to N
    reference_angular_velocity : (3,) array
        Angular velocity of R relative to N, expressed in N frame

    Returns
    -------
    beta_BR : (3,) array
        Attitude error of B relative to R in MRP form
    wB_BR : (3,) array
        Angular velocity tracking error of B relative to R, expressed in B frame
    """
    # Convert body MRP to DCM
    BN = MRP2DCM(measured_body_attitude_mrp)

    # Relative rotation from B to R
    RB = BN @ reference_attitude_DCM.T
    beta_BR = DCM2MRP(RB)  # convert relative rotation to MRP

    # Angular velocity tracking error
    wB_BR = measured_angular_velocity - BN @ reference_angular_velocity
    #wN_BR = BN.T @ measured_angular_velocity - reference_angular_velocity
    #wB_BR = BN @ wN_BR

    return beta_BR, wB_BR

def angle_between_two_vec(vec1, vec2):
    """
    Computes the angle between two vectors in degrees.

    Arguments:
        vec1, vec2 : np.array or list-like
    Returns:
        theta_deg : float -- angle between vectors in degrees
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

def getMissionMode(inertialPosLMO, inertialPosGMO, t):
    mode = INIT
    if(t < 1000):
        mode = INIT
    elif( inertialPosLMO[1] >= 0 and t >= 1000): 
        mode = SUN_MODE
    elif( angle_between_two_vec(inertialPosLMO, inertialPosGMO) < ANG_DIFF_FOR_COMMUNICATIONS):
        mode = COMMS_MODE
    else:
        mode = NADIR_MODE
        
    return mode

# Coordinate frame transformations
def Inertial2Hill(t):
    '''
    Computes the Direct Cosine Matrix from the Hill Frame (LMO) to the inertial frame (N)
    ----------
    Arguments:
        t {float} -- time from t_0, in seconds
    ----------
    Returns:
        (3,3) np.array -- Direct Cosine Matrix HN
    '''
    # Initial conditions for LMO
    theta_dot_LMO = np.sqrt(mu/r_LMO**3) #rad/sec
    angles_LMO = np.array([20,30,60])*np.pi/180 #Ω,i,θ(t_0), in radians
    angles_LMO += np.array([0,0, theta_dot_LMO*t]) #Ω,i,θ(t), in radians
    angles_LMO = wrap_angle_rad_2pi(angles_LMO) # + np.pi) % (2 * np.pi) - np.pi
    
    #Computing inertial np.arrays
    r_inertial_LMO, r_dot_inertial_LMO = orbitalElements2Inertial(r_LMO, angles_LMO)
    
    #Computing Hill Frame versors
    i_r = r_inertial_LMO/np.linalg.norm(r_inertial_LMO)
    i_h = np.cross(r_inertial_LMO, r_dot_inertial_LMO)/np.linalg.norm(\
                            np.cross(r_inertial_LMO, r_dot_inertial_LMO))
    i_theta = np.cross(i_h, i_r)
    NH = np.array([i_r, i_theta, i_h])
    HN = NH.T
    return normalize_dcm(HN)

def orbitalElements2Inertial(r, angles):
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
    n = np.sqrt(mu / r**3)
    v_perif = np.array([0, r*n, 0])

    rN = perif_to_ECI @ r_perif
    vN = perif_to_ECI @ v_perif

    return rN, vN


def init2Inertial():
    R = np.array([
        [1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1]
    ])
    return R

def Sun2Inertial():
    """
    Returns the transformation matrix from Sun-pointing reference frame (Rs) to the inertial frame (N) 
    
    The Rs-frame is defined as:
        Rs-frame = [-n1, n2 x (-n1), n2]
    where the Sun-pointing frame is stationary relative to N.

    Returns
    -------
    R : ndarray, shape (3,3)
        Rotation matrix from N-frame to Rs-frame
    """
    R = np.array([
        [-1, 0, 0],
        [ 0, 0, 1],
        [ 0, 1, 0]
    ])
    return R

def computeAngularVelocitySun():
    '''
    Computes the agular velocity with respect to inertial frame written in Inertial Reference Frame
    ----------
    Returns:
        (3,1) np.array -- Angular velocity w_RsN at time t
    '''    
    return np.array([0,0,0])

def Nadir2Inertial(t):
    '''
    Computes the Direct Cosine Matrix from the Nadir Reference Frame (Rn) to the inertial frame (N)
    ----------
    Arguments:
        t {float} -- time from t_0, in seconds
    ----------
    Returns:
        (3,3) np.array -- Direct Cosine Matrix RnN
    '''
    # Defining the versors from Rn in the Hill Frame (LMO) to construct the DCM [HRn]
    r_1 = np.array([-1,0,0])
    r_2 = np.array([0,1,0])
    r_3 = np.array([0,0,-1])
    DCM_HRn = np.array([r_1, r_2, r_3])
    
    # [RnN] = [RnH][HN]
    RnN = DCM_HRn @ Inertial2Hill(t).T
    return normalize_dcm(RnN)

def computeAngularVelocityNadir(t):
    '''
    Computes the agular velocity with respect to inertial frame written in Inertial Reference Frame
    ----------
    Arguments:
        t {float} -- time from t_0, in seconds
    ----------
    Returns:
        (3,1) np.array -- Angular velocity w_RnN at time t
    '''
    # Initial conditions for LMO
    theta_dot_LMO = np.sqrt(mu/r_LMO**3) #rad/sec   
    
    # w normally would be, in the H frame:
    w_HN_H = np.array([0,0,theta_dot_LMO])
    
    # Now we just compute that in the N frame
    w_HN_N =  Inertial2Hill(t).T @ w_HN_H
   
    return w_HN_N

def Comms2Inertial(t):
    '''
    Computes the Direct Cosine Matrix from the GMO-Pointing Reference Frame (Rc) to the inertial frame (N)
    ----------
    Arguments:
        t {float} -- time from t_0, in seconds
    ----------
    Returns:
        (3,3) np.array -- Direct Cosine Matrix RcN
    '''

    #LMO
    theta_dot_LMO = np.sqrt(mu/r_LMO**3) #rad/sec
    angles_LMO = np.array([20,30,60])*np.pi/180 #Ω,i,θ(t_0), in radians
    angles_LMO[2] = angles_LMO[2] + theta_dot_LMO*t #Ω,i,θ(t), in radians
    angles_LMO = wrap_angle_rad_2pi(angles_LMO)
    #angles_LMO = (angles_LMO + np.pi) % (2 * np.pi) - np.pi

    #GMO
    theta_dot_GMO = np.sqrt(mu/r_GMO**3) #rad/s
    angles_GMO = np.array([0,0,250])*np.pi/180 #Ω,i,θ(t_0), in radians
    angles_GMO += np.array([0,0, theta_dot_GMO*t]) #Ω,i,θ(t), in radians
    angles_GMO = wrap_angle_rad_2pi(angles_GMO)
    #angles_GMO = (angles_GMO + np.pi) % (2 * np.pi) - np.pi
    
    
    # Building delta_r
    r_GMO_inertial, v_GMO_inertial = orbitalElements2Inertial(r_GMO, angles_GMO)
    r_LMO_inertial, v_LMO_inertial = orbitalElements2Inertial(r_LMO, angles_LMO)
    delta_r = r_GMO_inertial - r_LMO_inertial
    
    #Rc frame
    n_3 = np.array([0,0,1])
    rc_1 = -delta_r/np.linalg.norm(delta_r)
    rc_2 = np.cross(delta_r, n_3)/np.linalg.norm(np.cross(delta_r, n_3))
    rc_3 = np.cross(rc_1, rc_2)
    
    DCM_RcN = np.array([rc_1, rc_2, rc_3]).T
    return DCM_RcN

def computeAngularVelocityComms(t):
    '''
    Computes the angular velocity from the GMO-Pointing Reference Frame (Rc) to the inertial frame (N) written in N
    ----------
    Arguments:
        t {float} -- time from t_0, in seconds
    ----------
    Returns:
        (3,1) np.array -- Angular velocity w_RcN at time t
    '''
    # d/dt(RcN) = (Rc(t+dt) - Rc(t))/dt
    RcN_dot = (Comms2Inertial(t+dt) - Comms2Inertial(t))/dt
    w_tilde = -Comms2Inertial(t).T @ RcN_dot
    w = np.array([-w_tilde[1][2],w_tilde[0][2],-w_tilde[0][1]])
    return w

# Dynamics
def tilde_matrix(v):
    """
    Returns the skew-symmetric (tilde) matrix of a 3-np.array v
    """
    #v = np.asarray(v).reshape(3)
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0]
    ])


def omega_dot(omega, I, u):
    """
    Compute the time derivative of angular velocity for a rigid body.
    Automatically handles 1D or 2D column np.array inputs.

    Parameters
    ----------
    omega : array-like, shape (3,) or (3,1)
        Angular velocity in body frame (rad/s)
    I : ndarray, shape (3,3)
        Inertia matrix of the body
    u : array-like, shape (3,) or (3,1)
        Control torque applied in body frame

    Returns
    -------
    w_dot : ndarray, shape (3,)
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

def dynamics(x, u, t):
    """
    Spacecraft dynamics using Modified Rodrigues Parameters (MRP)

    Parameters
    ----------
    x : ndarray, shape (6,)
        State np.array [sigma (3x1); omega_BN (3x1)] where
        sigma: MRP attitude
        omega_BN: angular velocity of B relative to N in B frame
    u : ndarray, shape (3,)
        Control torque np.array
    t : float
        Time (not used, included for ODE solver compatibility)

    Returns
    -------
    xdot : ndarray, shape (6,)
        Time derivative of state np.array
    """
    I_b_inv = np.linalg.inv(I_b)

    sigma = x[0:3]
    omega = x[3:6]
    
    sigma_n2 = np.dot(sigma.T, sigma)
   
    # MRP kinematics
    B = (1 - sigma_n2) * np.eye(3) + 2 * tilde_matrix(sigma) + 2 * np.outer(sigma, sigma.T)
    sigma_dot = 0.25 * B @ omega

    L = np.array([0,0,0])
    #skew = -np.cross(omega, I_b @ omega) + u + L
    
    # Angular momentum
    H = I_b @ omega  # shape (3,)
    cross_term = tilde_matrix(omega) @ H  # shape (3,)
    omega_dot = np.linalg.inv(I_b) @ (u - cross_term)  # shape (3,)
    
    # Combine into xdot
    xdot = np.hstack( (sigma_dot,omega_dot) )
    return xdot

def RK4(xdot_func, x_t, u_t, t, dt):
    """
    Fourth-order Runge-Kutta integration step.

    Parameters
    ----------
    xdot_func : callable
        Function computing the time derivative of the state: xdot = xdot_func(x, u, t)
    x_t : ndarray
        Current state np.array
    u_t : ndarray
        Control input at current time
    t : float
        Current time
    dt : float
        Time step

    Returns
    -------
    x_next : ndarray
        State np.array at time t + dt
    """
    
    k1 = xdot_func(x_t, u_t, t)
    k2 = xdot_func(x_t + k1 * dt / 2, u_t, t + dt / 2)
    k3 = xdot_func(x_t + k2 * dt / 2, u_t, t + dt / 2)
    k4 = xdot_func(x_t + k3 * dt, u_t, t + dt)

    x_next = x_t + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    return x_next

def animate_orbit_and_attitude(inertialPosLMO_history, ref_attitudes, meas_attitudes, mode_history, time_history, interval=1):
    """
    Animate 3D orbit and coordinate frames of reference with attitude rotation.

    Arguments:
        inertialPosLMO_history : list of 3D positions (Nx3 array) -- position history for the orbit
        ref_attitudes  : list of 3x3 np.array -- reference DCMs
        meas_attitudes : list of 3x3 np.array -- measured DCMs
        mode_history   : list of int           -- mode at each timestep
        time_history   : list of float         -- time at each timestep
        interval       : int                   -- ms between frames
    """
    # Define mode labels
    mode_labels = {0: "INIT", 1: "SUN_MODE", 2: "NADIR_MODE", 3: "COMMS_MODE"}

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Orbit data
    pos_array = np.array(inertialPosLMO_history)
    ax.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], 'b', label='Orbit')
    ax.scatter([0], [0], [0], color='red', s=80, label='Mars')  # Mars position at origin

    # Set axis limits (make sure they are large enough to include both the orbit and the attitude axes)
    max_range = np.max(np.abs(pos_array))  # Automatically scale axis limits based on the orbit's range
    margin = 1.2  # Add a 20% margin to the limits to ensure the attitude axes fit within the plot

    ax.set_xlim([-max_range * margin, max_range * margin])
    ax.set_ylim([-max_range * margin, max_range * margin])
    ax.set_zlim([-max_range * margin, max_range * margin])

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Orbit and Attitude Animation')
    ax.legend()
    ax.grid(True)
    ax.set_box_aspect([1, 1, 1])

    # Initialize plot lines for reference (faded) and measured (strong)
    ref_lines = [ax.plot([0, 0], [0, 0], [0, 0], color=c, alpha=0.3)[0] for c in ['r', 'g', 'b']]
    meas_lines = [ax.plot([0, 0], [0, 0], [0, 0], color=c, alpha=1.0, linewidth=2)[0] for c in ['r', 'g', 'b']]
    mode_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=14)
    time_text = ax.text2D(0.02, 0.90, "", transform=ax.transAxes, fontsize=12)

    # List to store black lines that will be removed each frame
    black_lines = []

    def update(frame):
        # Get current position from orbit path
        position = pos_array[frame]

        # Update the attitude frames based on the current position and attitude data
        R_ref = ref_attitudes[frame]
        R_meas = meas_attitudes[frame]

        # Scale factor for visibility of axes (adjust this to ensure the axes are visible but not too large)
        scale_factor = 1000*3  # Adjust this value for the proper visibility

        # Update reference axes (faded)
        for i, line in enumerate(ref_lines):
            line.set_data([position[0], position[0] + scale_factor * R_ref[0, i]], 
                          [position[1], position[1] + scale_factor * R_ref[1, i]])
            line.set_3d_properties([position[2], position[2] + scale_factor * R_ref[2, i]])

        # Update measured axes (strong)
        for i, line in enumerate(meas_lines):
            line.set_data([position[0], position[0] + scale_factor * R_meas[0, i]], 
                          [position[1], position[1] + scale_factor * R_meas[1, i]])
            line.set_3d_properties([position[2], position[2] + scale_factor * R_meas[2, i]])

        # Remove previous black lines
        for line in black_lines:
            line.remove()
        black_lines.clear()  # Clear the list of black lines

        # Direction from the center (attitude frame) to Mars (origin)
        direction_to_mars = np.array([0, 0, 0]) - position  # Vector pointing from attitude frame to Mars (origin)

        # Scale the line to be 1.5 times the length of the reference frame's axes (scaled direction)
        scaled_direction = 1.5 * direction_to_mars

        # Draw the black line from the center (position) to the scaled direction towards Mars
        black_line = ax.plot([position[0], position[0] + scaled_direction[0]], 
                             [position[1], position[1] + scaled_direction[1]],
                             [position[2], position[2] + scaled_direction[2]], 
                             color='black', linewidth=0.5)

        # Add the black line to the list for future removal
        black_lines.append(black_line[0])

        # Update mode text
        mode_text.set_text(f"Mode: {mode_labels.get(mode_history[frame], 'UNKNOWN')}")
        time_text.set_text(f"Time: {time_history[frame]:.1f} s")

        return ref_lines + meas_lines + [mode_text, time_text]

    ani = FuncAnimation(fig, update, frames=len(ref_attitudes), interval=interval, blit=False)
    plt.show()
    return ani


def animate_attitude(ref_attitudes, meas_attitudes, mode_history, time_history, interval=1):
    """
    Animate 3D coordinate frames of reference and measured attitude.

    Arguments:
        ref_attitudes  : list of 3x3 np.array -- reference DCMs
        meas_attitudes : list of 3x3 np.array -- measured DCMs
        mode_history   : list of int           -- mode at each timestep
        interval       : int                   -- ms between frames
    """
    
    # Define mode labels
    mode_labels = {0: "INIT", 1: "SUN_MODE", 2: "NADIR_MODE", 3: "COMMS_MODE"}
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Attitude Animation')

    # Initialize plot lines for reference (faded) and measured (strong)
    ref_lines = [ax.plot([0,0],[0,0],[0,0], color=c, alpha=0.3)[0] for c in ['r','g','b']]
    meas_lines = [ax.plot([0,0],[0,0],[0,0], color=c, alpha=1.0, linewidth=2)[0] for c in ['r','g','b']]
    mode_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=14)
    time_text = ax.text2D(0.02, 0.90, "", transform=ax.transAxes, fontsize=12)
    
    def update(frame):
        R_ref = ref_attitudes[frame]
        R_meas = meas_attitudes[frame]
        
        origin = np.zeros(3)
        
        # Update reference axes (faded)
        for i, line in enumerate(ref_lines):
            line.set_data([origin[0], R_ref[0,i]], [origin[1], R_ref[1,i]])
            line.set_3d_properties([origin[2], R_ref[2,i]])
        
        # Update measured axes (strong)
        for i, line in enumerate(meas_lines):
            line.set_data([origin[0], R_meas[0,i]], [origin[1], R_meas[1,i]])
            line.set_3d_properties([origin[2], R_meas[2,i]])
        
        # Update mode text
        mode_text.set_text(f"Mode: {mode_labels.get(mode_history[frame], 'UNKNOWN')}")
        time_text.set_text(f"Time: {time_history[frame]:.1f} s")
        
        return ref_lines + meas_lines + [mode_text]
    
    ani = FuncAnimation(fig, update, frames=len(ref_attitudes), interval=interval, blit=False)
    plt.show()
    return ani

def main():
    print("SIMULATING")
    # time parameters
    time = 0

   # Initial conditions
    #LMO
    theta_dot_LMO = np.sqrt(mu/r_LMO**3) #rad/sec
    angles_LMO = np.array([20,30,60])*np.pi/180 #Ω,i,θ(t_0), in radians
    inertialPosLMO, inertialVelLMO = orbitalElements2Inertial(r_LMO, angles_LMO)
        
    #GMO
    theta_dot_GMO = np.sqrt(mu/r_GMO**3) #rad/s
    angles_GMO = np.array([0,0,250])*np.pi/180 #Ω,i,θ(t_0), in radians
    inertialPosGMO, inertialVelGMO = orbitalElements2Inertial(r_GMO, angles_GMO)
    
    #Spacecraft
    sigma_BN = np.array([0.3, -0.4, 0.5])
    w_BN_B = np.array([1.0, 1.75, -2.2])*np.pi/180 # radians/s

    # arrays to store data for visualization and debugging purposes
    mode = INIT
    mode_history = [mode]
    attitudeError_history = [ np.array([0,0,0]) ] 
    angularVelError_history = [ np.array([0,0,0]) ]
    reference_sigma_history = [sigma_BN]
    reference_angularVel_history = [w_BN_B]
    measured_sigma_history = [sigma_BN]
    measured_angularVel_history = [w_BN_B]

    reference_attitude_DCM_history = [MRP2DCM(sigma_BN)]
    measured_DCM_history = [(MRP2DCM(sigma_BN))]
    attitudeError_DCM_history= [MRP2DCM(sigma_BN)]

    control_history = [np.array([0,0,0])]
    inertialPosLMO_history = [ inertialPosLMO ]
    inertialPosGMO_history = [ inertialPosGMO ]

    time_history = [0]
    
    while(time < simTime):

        #if(time % 100 == 0):
         #  print("TIME = "+ str(time) )
        
        # Get measured state from sensors
        measured_body_attitude_mrp = sigma_BN
        measured_angular_velocity = w_BN_B
        

        # Determine the mission mode
        mode = getMissionMode(inertialPosLMO, inertialPosGMO, time)
        if(mode==INIT):
            reference_attitude_DCM = init2Inertial()
            reference_angular_velocity = computeAngularVelocitySun()
            attitudeError, angularVelError = calculateError(measured_body_attitude_mrp,
                measured_angular_velocity,
                reference_attitude_DCM,
                reference_angular_velocity)

        elif(mode==SUN_MODE): # Recharge batteries
            reference_attitude_DCM = Sun2Inertial()
            reference_angular_velocity = computeAngularVelocitySun()
            attitudeError, angularVelError = calculateError(measured_body_attitude_mrp,
                measured_angular_velocity,
                reference_attitude_DCM,
                reference_angular_velocity)
        
        elif(mode==NADIR_MODE): # Science
            reference_attitude_DCM = Nadir2Inertial(time)
            reference_angular_velocity = computeAngularVelocityNadir(time)
            attitudeError, angularVelError = calculateError(measured_body_attitude_mrp,
                measured_angular_velocity,
                reference_attitude_DCM,
                reference_angular_velocity)

        else: # Communicate with mother ship
            reference_attitude_DCM = Comms2Inertial(time)
            reference_angular_velocity = computeAngularVelocityComms(time)
            attitudeError, angularVelError = calculateError(measured_body_attitude_mrp,
                measured_angular_velocity,
                reference_attitude_DCM,
                reference_angular_velocity)
            
        # Calculate Control Input
        P = np.max(I_b * (2/120))
        K = (P**2) / I_b[1, 1]
       
        u = -K * attitudeError - P * angularVelError
        control_history.append(u)

        # Propagate Spacecraft Dynamics
        x_t0 = np.hstack((sigma_BN, w_BN_B))
        x_t1 = RK4(dynamics, x_t0, u, time, dt)
        sigma_BN = x_t1[:3]
        w_BN_B = x_t1[-3:] 

        # Store data for visualization and debugging
        mode_history.append(mode)
        attitudeError_history.append(attitudeError)
        attitudeError_DCM_history.append(MRP2DCM(attitudeError))
        angularVelError_history.append(angularVelError)
        
        reference_attitude_DCM_history.append(reference_attitude_DCM)
        reference_sigma_history.append( DCM2MRP(reference_attitude_DCM))
        reference_angularVel_history.append(reference_angular_velocity)
        measured_sigma_history.append(sigma_BN)
        measured_DCM_history.append(MRP2DCM(sigma_BN))
        measured_angularVel_history.append(w_BN_B)

        # Propagate time
        time = time + dt
        time_history.append(time)

        # Propagate dynamics (theta - mean anomaly)
        angles_LMO += np.array([0,0, theta_dot_LMO*dt]) #Ω,i,θ(t), in radians
        angles_GMO += np.array([0,0, theta_dot_GMO*dt]) #Ω,i,θ(t), in radians
        angles_LMO = wrap_angle_rad_2pi(angles_LMO)
        angles_GMO = wrap_angle_rad_2pi(angles_GMO)

        #angles_LMO = (angles_LMO + np.pi) % (2 * np.pi) - np.pi
        #angles_GMO = (angles_GMO + np.pi) % (2 * np.pi) - np.pi

        inertialPosLMO, inertialVelLMO = orbitalElements2Inertial(r_LMO, angles_LMO)
        inertialPosGMO, inertialVelGMO = orbitalElements2Inertial(r_GMO, angles_GMO)

        inertialPosLMO_history.append(inertialPosLMO)
        inertialPosGMO_history.append(inertialPosGMO)

    if(SHOW_ATTITUDE):
        animate_orbit_and_attitude(inertialPosLMO_history, reference_attitude_DCM_history,measured_DCM_history, mode_history, time_history)
    
    # Plot
    if(PLOT_OFFLINE):

        # Plot
        plt.figure(figsize=(10,5))

        # MRP norm
        plt.subplot(2,1,1)
        plt.plot(time_plot, np.linalg.norm(attitudeError_history, axis=1))
        plt.title("MRP Norm over Time")
        plt.ylabel("||σ||")
        plt.grid(True)

        # DCM error trace
        plt.subplot(2,1,2)
        race_Rerr = [np.trace(dcm_err) for dcm_err in attitudeError_DCM_history]
        plt.plot(time_plot, race_Rerr)
        plt.title("Trace of Error DCM over Time")
        plt.xlabel("Time [s]")
        plt.ylabel("trace(R_ref^T R_meas)")
        plt.grid(True)

        plt.tight_layout()

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        pos_array = np.array(inertialPosLMO_history)
        print(pos_array.shape)
        ax.plot(pos_array[:,0], pos_array[:,1], pos_array[:,2], 'b', label='Orbit')
        ax.scatter([0], [0], [0], color='red', s=80, label='Mars')
        # Formatting
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title('3D Orbit Visualization')
        ax.legend()
        ax.grid(True)
        ax.set_box_aspect([1,1,1])

        plt.figure(figsize=(10,6))
        plt.subplot(2,1,1)
        plt.plot(time_plot, measured_sigma_history)
        plt.title("MRP Attitude σ_BN vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("σ")
        plt.legend(['σ1','σ2','σ3'])
        plt.subplot(2,1,2)
        plt.plot(time_plot, measured_angularVel_history)
        plt.title("Angular Rate ω_BN vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("ω (rad/s)")
        plt.legend(['ω1','ω2','ω3'])
        plt.tight_layout()

        plt.figure(figsize=(10,6))
        plt.subplot(2,1,1)
        plt.plot(time_plot, attitudeError_history)
        plt.title("MRP σ_BR Error vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("σ")
        plt.legend(['σ1','σ2','σ3'])
        plt.subplot(2,1,2)
        plt.plot(time_plot, angularVelError_history)
        plt.title("Angular Rate Error ω_BR vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("ω (rad/s)")
        plt.legend(['ω1','ω2','ω3'])
        plt.tight_layout()

        plt.figure()
        plt.plot(time_plot, mode_history, drawstyle='steps-post')  # optional: step plot for discrete modes
        plt.title("Vehicle Mode History vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Mode")
        plt.grid(True)
        # Set y-ticks with mode labels
        plt.yticks(
            [INIT, SUN_MODE, NADIR_MODE, COMMS_MODE], 
            ["INIT", "SUN_MODE", "NADIR_MODE", "COMMS_MODE"]
        )
    
        plt.figure(figsize=(8,4))
        plt.plot(time_plot, control_history)
        plt.title("Control Torque over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Torque (N·m)")
        plt.legend(['τ1','τ2','τ3'])
        plt.tight_layout()

        plt.show()

    print("SIMULATION ENDED")
    if(ANIM_3D):
        print("ANIMATING")
        ani = animate(inertialPosLMO_history, inertialPosGMO_history, measured_DCM_history, attitudeError_DCM_history, mode_history, dt)
        print("ANIMATION ENDED")
        print("END")

    # print("END")
# __name__
if __name__=="__main__":
    main()