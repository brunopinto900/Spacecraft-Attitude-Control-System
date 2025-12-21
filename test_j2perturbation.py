import numpy as np
import matplotlib.pyplot as plt
from config import * # Import the constants from config.py

def j2_orbital_element_rates(a, inclination, J2=1.08263e-3):
    a = R_planet + h
    incl = inclination
    n = np.sqrt(mu / a**3)  # rad/s

    dRAAN_dt = -1.5 * J2 * (R_planet / a)**2 * n * np.cos(incl)  # rad/s
    domega_dt = 0.75 * J2 * (R_planet / a)**2 * n * (5*np.cos(incl)**2 - 1)  # rad/s

    # Convert rad/s → deg/day
    rad2deg = 180/np.pi
    sec2day = 86400
    dRAAN_dt_deg = dRAAN_dt * rad2deg * sec2day
    domega_dt_deg = domega_dt * rad2deg * sec2day

    return dRAAN_dt_deg, domega_dt_deg

# CubeSat parameters
a = 6378.137 + 400  # km
incl_deg = 51.6      # typical LEO

# Compute drift rates
dRAAN_dt, domega_dt = j2_orbital_element_rates(a, incl_deg)

# Time vector: 30 days
days = np.arange(0, 31, 1)
RAAN = dRAAN_dt * days
omega = domega_dt * days

# Plot
plt.figure(figsize=(10,5))
plt.plot(days, RAAN, label="RAAN (deg)")
plt.plot(days, omega, label="Arg. of Perigee (deg)")
plt.xlabel("Time [days]")
plt.ylabel("Angle [deg]")
plt.title("J2-Induced Secular Drift of RAAN and Argument of Perigee")
plt.grid(True)
plt.legend()
plt.show()
