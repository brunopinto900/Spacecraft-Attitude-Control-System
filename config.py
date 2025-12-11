import numpy as np

# ------------------------------
# Planet and Orbit Parameters
# ------------------------------
R_planet = 3396.19  # Radius of Mars in kilometers (km)
h = 400  # Altitude of the Low Mars Orbit (LMO) in kilometers (km)
mu = 42828.3  # Gravitational parameter for Mars (in km^3/s^2)
r_LMO = R_planet + h  # Radius of the Low Mars Orbit (LMO) in kilometers (km)
r_GMO = 20424.2  # Radius of the Geostationary Mars Orbit (GMO) in kilometers (km)

# ------------------------------
# Display Settings
# ------------------------------
PLOT_OFFLINE = False  # Set to True to disable interactive plotting (offline mode)
ANIM_3D = True  # Set to True to enable 3D animation
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
# The values are given in units of kg*m^3.
# Here we assume a simple diagonal inertia matrix, representing the principal moments of inertia about each axis.
I_b = np.diag([10.0, 5.0, 7.5])  # Inertia matrix in kg*m^3 (diagonal form, for simplicity)
