import numpy as np

# Functions to compute total mass and inertia
def skew(v):
    """Return skew-symmetric matrix for vector v."""
    v = np.asarray(v).reshape(3)
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def Rscrew(n):
    """
    Equivalent to MATLAB Rscrew(n): rotation matrix that aligns wheel's spin axis with direction n.
    Assumes n is a 3×1 vector.
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
    # This matches MATLAB: T = Rscrew(n)
    return np.column_stack((n, y, z))


def computeInertiaReactionWheels():
    # Offsets from spacecraft center of mass (4 mm)
    r1 = np.array([4, 0, 0]) / 1000
    r2 = np.array([0, 4, 0]) / 1000
    r3 = np.array([0, 0, 4]) / 1000

    # Inertia of the reaction wheel (disk)
    Idisk = (1/12) * (3*radius_RW**2 + height_RW**2)
    IrR = mass_RW * np.array([
        [(1/2)*radius_RW**2, 0, 0],
        [0, Idisk, 0],
        [0, 0, Idisk]
    ])

    # Maximum angular acceleration
    maxAlpha = maxTorque / IrR[0, 0]      # rad/s^2

    # Power/torque ratios
    Power2Alpha = peak_power / maxAlpha
    Amps2Alpha = Power2Alpha / dc_voltage

    # Rotation matrices from RW frame to body frame
    T1 = Rscrew(e1_RW)
    T2 = Rscrew(e2_RW)
    T3 = Rscrew(e3_RW)

    # Inertia tensors of wheels in body frame
    Ir1B = T1.T @ IrR @ T1
    Ir2B = T2.T @ IrR @ T2
    Ir3B = T3.T @ IrR @ T3

    # Control Allocation Matrix J
    J = np.column_stack((Ir1B @ e1_RW, Ir2B @ e2_RW, Ir3B @ e3_RW))

    # Inverse (pseudo-inverse for >3 wheels):
    Jinv = J.T @ np.linalg.inv(J @ J.T)

    # Parallel-axis theorem: wheel inertia about satellite CG
    I_RW1Bcg = Ir1B + mass_RW * (skew(r1).T @ skew(r1))
    I_RW2Bcg = Ir2B + mass_RW * (skew(r2).T @ skew(r2))
    I_RW3Bcg = Ir3B + mass_RW * (skew(r3).T @ skew(r3))

    return I_RW1Bcg, I_RW2Bcg, I_RW3Bcg, Ir1B, Ir2B, Ir3B

# ------------------------------
# Planet and Orbit Parameters
# ------------------------------
R_planet = 3396.19  # Radius of Mars in kilometers (km)
h = 400  # Altitude of the Low Mars Orbit (LMO) in kilometers (km)
mu = 42828.3  # Gravitational parameter for Mars (in km^3/s^2)
r_LMO = R_planet + h  # Radius of the Low Mars Orbit (LMO) in kilometers (km)
r_GMO = 20424.2  # Radius of the Geostationary Mars Orbit (GMO) in kilometers (km)

mean_anomaly = 60 # degrees
inclination = 30 # degrees
RAAN = 20 # degrees

# ------------------------------
# Display Settings
# ------------------------------
PLOT_OFFLINE = True  # Set to True to disable interactive plotting (offline mode)
ANIM_3D = False  # Set to True to enable 3D animation
SHOW_ATTITUDE = False  # Set to True to show the satellite's attitude (orientation)

# ------------------------------
# Mission Modes
# ------------------------------
# Defining the different operational modes for the satellite:
INIT = 0  # Initialization mode (e.g., starting conditions or booting)
SUN_MODE = 1  # Sun-pointing mode (satellite faces the Sun for power or thermal reasons)
NADIR_MODE = 2  # Nadir-pointing mode (satellite faces downwards to the planet's surface)
COMMS_MODE = 3  # Communications mode (satellite is oriented to communicate with ground stations or other satellites)

# ------------------------------
# Mission Parameters
# ------------------------------
# This parameter defines the angular difference that needs to be met for communication to be possible
# between the Low Mars Orbit (LMO) and Geostationary Mars Orbit (GMO) satellites. The angular difference 
# must be less than the specified threshold for communications to be possible.
ANG_DIFF_FOR_COMMUNICATIONS = 35  # Maximum allowable angular difference for satellite communications (in degrees)

# ------------------------------
# Satellite Parameters
# ------------------------------
# Satellite's inertia matrix (I_b) is a 3x3 matrix that describes how the satellite resists rotational motion.
# The values are given in units of kg*m^2.
# Here we assume a simple diagonal inertia matrix, representing the principal moments of inertia about each axis.

Lx = 20/100 #m length x-direction
Ly = 10/100 #m length y-direction
Lz = 15/100 #m length z-direction
mass_sat = 2.6 #kg
Ixx = (1/12) * mass_sat * (Ly**2 + Lz**2)
Iyy = (1/12) * mass_sat * (Lx**2 + Lz**2)
Izz = (1/12) * mass_sat * (Lx**2 + Ly**2)
I_sat = np.diag([Ixx, Iyy, Izz])
SA = 2 * (Lx*Ly + Lx*Lz + Ly*Lz) # m^2 Surface Area

# Actuators parameters
### Reaction Wheel
mass_RW = 0.13 #Kg
radius_RW = 42/1000 #m
height_RW = 19/1000 #m
# Maximum Speed
rpm = 8000
maxSpeed = rpm * 2*np.pi / 60   # rad/s
# Maximum Torque
maxTorque = 0.004     # N·m
# Power parameters
dc_voltage = 5.0      # Volts
peak_power = 3.25     # Watts
# RW axes
e1_RW = np.array([1, 0, 0])
e2_RW = np.array([0, 1, 0])
e3_RW = np.array([0, 0, 1])
### Reaction Wheel

# Total mass and inertia
mass = mass_sat + 3*mass_RW
I_RW1Bcg, I_RW2Bcg, I_RW3Bcg, Ir1B, Ir2B, Ir3B = computeInertiaReactionWheels() # inertia matrix of RW in body frame
I_RW = I_RW1Bcg + I_RW2Bcg + I_RW3Bcg
I_b = I_sat + I_RW

# Environmental Parameters
C_d = 2.2 # Darg coefficient

