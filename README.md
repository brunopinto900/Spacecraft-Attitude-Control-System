# Mars Satellite Simulation

This repository contains a simulation of a satellite in low orbit around a planet. It models the behavior of the satellite and includes functionality for computing orbital dynamics, satellite attitude, and visualizing the satellite's behavior in different operational modes.

The satellite performs three main mission modes:
- **Nadir Mode**: The satellite points toward the planet (nadir pointing).
- **Sun Mode**: The satellite points toward the Sun for solar energy collection.
- **Comms Mode**: The satellite aligns to communicate with the Geostationary Mars Orbit (GMO) satellite.

The satellite’s reference attitudes are provided by Direction Cosine Matrices (DCMs), and the attitudes are represented using Modified Rodrigues Parameters (MRP). The control law used for attitude stabilization is a Proportional-Derivative (PD) controller, which ensures precise tracking of the reference attitudes in all mission modes.

[![Watch the video](https://github.com/brunopinto900/Spacecraft-Attitude-Control-System/blob/main/media/mars_satellite_medium.mp4))


## Features
- **Orbital Mechanics**: Calculates orbital elements and transforms them between inertial and Hill frames.
- **Satellite Attitude**: Computes and displays satellite attitude using Direction Cosine Matrices (DCM) and Modified Rodrigues Parameters (MRP).
- **Mission Modes**: Allows switching between different operational modes (e.g., Sun-pointing, Nadir-pointing, etc.).
- **Visualization**: Supports 2D/3D visualizations of the satellite’s behavior and attitude.
- **Configurable Settings**: Easily configurable parameters for mission setup and visualization preferences.

## Tests
This project includes a set of tests to ensure that the code works as expected. The tests cover the following functionality:

- **Orthonormality**: Verifies that the transformation matrices are orthogonal.
- **Axes Definitions**: Validates the satellite's orientation in the Hill frame.
- **Continuity**: Ensures the continuity of the transformation matrices over time.
- **Inverse Transform**: Verifies that the inverse transformation produces correct results.
- **Angular Rate**: Ensures that the angular rate is computed correctly for the satellite’s motion.

## Running the Tests
To run the tests, write in the terminal: "pytest -v"

## Running the Script
To run the main simulation script, execute the following command in the terminal: python3 main.py

## Configure Simulation
The configuration file config.py contains key parameters for the simulation. These parameters control various aspects of the satellite's behavior, orbit, and visualizations. Here's a breakdown of each parameter:

### Planet and Orbit Parameters

- **R_planet**: Radius of the Planet (in kilometers). Default is 3396.19 km (Mars).
- **h**: Altitude of the Low Mars Orbit (LMO) in kilometers. Default is 400 km.
- **mu**: Gravitational parameter for the Planet (in km³/s²). Default is 42828.3 km³/s² (Mars).
- **r_LMO**: The total radius of the Low Orbit (LMO), which is R_planet + h. Default is 3796.19 km (Mars)
- **r_GMO**: The radius of the Geostationary Orbit (GMO), which is 20424.2 km (Mars)

### Display Settings

These settings control how the simulation is visualized:

- **PLOT_OFFLINE**: If set to True, plots offline data including satellite attitude, angular velocity error, attitude and angular velocity references, and control effort. Default is False. This mode disables interactive plotting and allows plotting in offline mode, such as for generating reports or analyzing results after the simulation runs.
- **ANIM_3D**: If set to True, enables 3D animation of the satellite's behavior. Default is False.
- **SHOW_ATTITUDE**: If set to True, shows the satellite’s attitude (its orientation or pointing direction). Default is True.

#### Note: Only one visualization option can be True at any time. You should choose either ANIM_3D for 3D animation or SHOW_ATTITUDE for attitude visualization, but not both simultaneously. The visualization settings are mutually exclusive.
