"""
Main simulation script for spacecraft attitude control system.

This script runs the spacecraft attitude control simulation, integrating
orbital dynamics, attitude control, and environmental perturbations.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import configuration and constants
from config import *

# Import modules
from math_utils import angle_between_two_vec, wrap_angle_rad_2pi
from attitude_transformations import MRP2DCM, DCM2MRP
from orbital_mechanics import orbitalElements2Inertial
from reference_frames import (init2Inertial, Sun2Inertial, Nadir2Inertial, Comms2Inertial,
                               computeAngularVelocitySun, computeAngularVelocityNadir, computeAngularVelocityComms)
from control import calculateError, calculateControl
from perturbations import calculateTorqueDisturbances, calculateForceDisturbances
from dynamics import dynamics, RK4, rk4_gravity_step
from visualization import animate, animate_orbit_and_attitude


# ============================================================
# Mission Mode Selection
# ============================================================

def getMissionMode(inertialPosLMO, inertialPosGMO, t):
    """
    Determine the current mission mode based on position and time.
    
    Parameters
    ----------
    inertialPosLMO : array-like
        LMO satellite position in inertial frame
    inertialPosGMO : array-like
        GMO satellite position in inertial frame
    t : float
        Current simulation time
        
    Returns
    -------
    int
        Current mission mode
    """
    mode = INIT
    if inertialPosLMO[1] >= 0:
        mode = SUN_MODE
    elif angle_between_two_vec(inertialPosLMO, inertialPosGMO) < ANG_DIFF_FOR_COMMUNICATIONS:
        mode = COMMS_MODE
    else:
        mode = NADIR_MODE

    return mode


# ============================================================
# Main Simulation Function
# ============================================================

def main():
    """Run the spacecraft attitude control simulation."""
    print("SIMULATING")
    
    # ========================================
    # Simulation Time Parameters
    # ========================================
    epoch = datetime(2025, 1, 1, 0, 0, 0)  # arbitrary fixed date
    orbitRadius = R_planet + h
    orbitPeriod = 2 * np.pi * np.sqrt(orbitRadius**3 / mu)  # seconds
    numberOfOrbits = 1
    simTime = numberOfOrbits * orbitPeriod
    print("SIM TIME (1 ORBIT)")
    print(simTime)
    
    nPoints = simTime
    time_plot = np.linspace(0, nPoints, int(simTime/dt)+2)
    print(time_plot.shape)

    # ========================================
    # Initial Conditions
    # ========================================
    time = 0

    # LMO
    theta_dot_LMO = np.sqrt(mu/r_LMO**3)  # rad/sec
    angles_LMO = np.array([RAAN, inclination, mean_anomaly]) * np.pi/180  # radians
    inertialPosLMO, inertialVelLMO = orbitalElements2Inertial(r_LMO, angles_LMO)
    inertialPosLMO_noDrag = inertialPosLMO.copy()
    inertialVelLMO_noDrag = inertialVelLMO.copy()
        
    # GMO
    theta_dot_GMO = np.sqrt(mu/r_GMO**3)  # rad/s
    angles_GMO = np.array([0, 0, 250]) * np.pi/180  # radians
    inertialPosGMO, inertialVelGMO = orbitalElements2Inertial(r_GMO, angles_GMO)
    
    # Spacecraft attitude and angular velocity
    sigma_BN = np.array([0.3, -0.4, 0.5])
    w_BN_B = np.array([1.0, 1.75, -2.2]) * np.pi/180  # radians/s
    omega_RW = np.array([0, 0, 0])  # Initial RW angular speed (zero initially)
 
    # ========================================
    # Data Storage Arrays
    # ========================================
    mode = INIT
    torqueDisturbances_history = [np.array([0, 0, 0])]
    forceDisturbances_history = [np.array([0, 0, 0])]
    mode_history = [mode]
    attitudeError_history = [np.array([0, 0, 0])]
    angularVelError_history = [np.array([0, 0, 0])]
    reference_sigma_history = [sigma_BN]
    reference_angularVel_history = [w_BN_B]
    measured_sigma_history = [sigma_BN]
    measured_angularVel_history = [w_BN_B]

    reference_attitude_DCM_history = [MRP2DCM(sigma_BN)]
    measured_DCM_history = [MRP2DCM(sigma_BN)]
    attitudeError_DCM_history = [MRP2DCM(sigma_BN)]

    control_history = [np.array([0, 0, 0])]
    inertialPosLMO_history = [inertialPosLMO]
    inertialPosGMO_history = [inertialPosGMO]
    inertialPosLMO_noDrag_history = [inertialPosLMO_noDrag]

    accelRW_history = [np.array([0, 0, 0])]
    omega_RW_history = [np.array([0, 0, 0])]

    time_history = [0]
    
    # ========================================
    # Main Simulation Loop
    # ========================================
    while time < simTime:
        # Get measured state from sensors
        measured_body_attitude_mrp = sigma_BN
        measured_angular_velocity = w_BN_B
        
        # Determine the mission mode
        mode = getMissionMode(inertialPosLMO, inertialPosGMO, time)
        
        # Get reference attitude and angular velocity based on mode
        if mode == INIT:
            reference_attitude_DCM = init2Inertial()
            reference_angular_velocity = computeAngularVelocitySun()
        elif mode == SUN_MODE:  # Recharge batteries
            reference_attitude_DCM = Sun2Inertial()
            reference_angular_velocity = computeAngularVelocitySun()
        elif mode == NADIR_MODE:  # Science
            reference_attitude_DCM = Nadir2Inertial(time)
            reference_angular_velocity = computeAngularVelocityNadir(time)
        else:  # COMMS_MODE - Communicate with mother ship
            reference_attitude_DCM = Comms2Inertial(time)
            reference_angular_velocity = computeAngularVelocityComms(time)
        
        # Calculate tracking errors
        attitudeError, angularVelError = calculateError(
            measured_body_attitude_mrp,
            measured_angular_velocity,
            reference_attitude_DCM,
            reference_angular_velocity
        )
            
        # Calculate Control Input
        u, accelRW = calculateControl(omega_RW, attitudeError, angularVelError)
        omega_RW = omega_RW + accelRW*dt
        
        # Calculate disturbances
        dist_torque = calculateTorqueDisturbances(sigma_BN, inertialPosLMO, inertialVelLMO)
        dist_force = calculateForceDisturbances(sigma_BN, inertialPosLMO, inertialVelLMO)
        
        # Propagate Spacecraft Attitude Dynamics
        x_t0 = np.hstack((sigma_BN, w_BN_B))
        x_t1 = RK4(dynamics, x_t0, dist_torque, u, omega_RW, time, dt)
        sigma_BN = x_t1[:3]
        w_BN_B = x_t1[-3:]

        # Propagate time
        time = time + dt

        # Propagate orbital dynamics (theta - mean anomaly)
        angles_LMO += np.array([0, 0, theta_dot_LMO*dt])  # Ω, i, θ(t), in radians
        angles_GMO += np.array([0, 0, theta_dot_GMO*dt])  # Ω, i, θ(t), in radians
        angles_LMO = wrap_angle_rad_2pi(angles_LMO)
        angles_GMO = wrap_angle_rad_2pi(angles_GMO)

        # Update GMO position
        inertialPosGMO, inertialVelGMO = orbitalElements2Inertial(r_GMO, angles_GMO)

        # Propagate LMO orbital position with and without drag
        inertialPosLMO_noDrag, inertialVelLMO_noDrag = rk4_gravity_step(
            np.array([0, 0, 0]), inertialPosLMO_noDrag, inertialVelLMO_noDrag, dt, mu)
        inertialPosLMO, inertialVelLMO = rk4_gravity_step(
            dist_force/mass, inertialPosLMO, inertialVelLMO, dt, mu)
        
        # Store data for visualization (only last orbit)
        if time >= (orbitPeriod*(numberOfOrbits-1)):
            time_history.append(time)
            control_history.append(u)
            torqueDisturbances_history.append(dist_torque)
            forceDisturbances_history.append(dist_force)
            mode_history.append(mode)
            attitudeError_history.append(attitudeError)
            attitudeError_DCM_history.append(MRP2DCM(attitudeError))
            angularVelError_history.append(angularVelError)
            reference_attitude_DCM_history.append(reference_attitude_DCM)
            reference_sigma_history.append(DCM2MRP(reference_attitude_DCM))
            reference_angularVel_history.append(reference_angular_velocity)
            measured_sigma_history.append(sigma_BN)
            measured_DCM_history.append(MRP2DCM(sigma_BN))
            measured_angularVel_history.append(w_BN_B)
            inertialPosLMO_history.append(inertialPosLMO)
            inertialPosGMO_history.append(inertialPosGMO)
            inertialPosLMO_noDrag_history.append(inertialPosLMO_noDrag)
            accelRW_history.append(accelRW)
            omega_RW_history.append(omega_RW)
    
    # ========================================
    # Visualization
    # ========================================
    if SHOW_ATTITUDE:
        animate_orbit_and_attitude(
            inertialPosLMO_history, 
            reference_attitude_DCM_history,
            measured_DCM_history, 
            mode_history, 
            time_history
        )
    
    # Plot results
    if PLOT_OFFLINE:
        # Reaction Wheel Performance
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(time_history, accelRW_history)
        plt.title("RW Angular Accel")
        plt.ylabel("rad/s^2")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(time_history, omega_RW_history)
        plt.title("RW Angular Vel (Omega)")
        plt.xlabel("Time [s]")
        plt.ylabel("rad/s")
        plt.grid(True)
        plt.tight_layout()

        # Attitude Error
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(time_history, np.linalg.norm(attitudeError_history, axis=1))
        plt.title("MRP Norm over Time")
        plt.ylabel("||σ||")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        trace_Rerr = [np.trace(dcm_err) for dcm_err in attitudeError_DCM_history]
        plt.plot(time_history, trace_Rerr)
        plt.title("Trace of Error DCM over Time")
        plt.xlabel("Time [s]")
        plt.ylabel("trace(R_ref^T R_meas)")
        plt.grid(True)
        plt.tight_layout()

        # Disturbance Forces
        fig = plt.figure(figsize=(8, 8))
        plt.plot(time_history, forceDisturbances_history)
        plt.title("Disturbance Force over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.legend(['f1', 'f2', 'f3'])
        plt.tight_layout()
        
        # 3D Orbit
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        pos_array = np.array(inertialPosLMO_history)
        pos_array_noDrag = np.array(inertialPosLMO_noDrag_history)
        ax.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], 'b', label='Orbit')
        ax.plot(pos_array_noDrag[:, 0], pos_array_noDrag[:, 1], pos_array_noDrag[:, 2], 'g--', label='Orbit (No drag)')
        ax.scatter([0], [0], [0], color='red', s=80, label='Mars')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title('3D Orbit Visualization')
        ax.legend()
        ax.grid(True)
        ax.set_box_aspect([1, 1, 1])

        # Measured Attitude
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(time_history, measured_sigma_history)
        plt.title("MRP Attitude σ_BN vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("σ")
        plt.legend(['σ1', 'σ2', 'σ3'])
        
        plt.subplot(2, 1, 2)
        plt.plot(time_history, measured_angularVel_history)
        plt.title("Angular Rate ω_BN vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("ω (rad/s)")
        plt.legend(['ω1', 'ω2', 'ω3'])
        plt.tight_layout()

        # Tracking Errors
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(time_history, attitudeError_history)
        plt.title("MRP σ_BR Error vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("σ")
        plt.legend(['σ1', 'σ2', 'σ3'])
        
        plt.subplot(2, 1, 2)
        plt.plot(time_history, angularVelError_history)
        plt.title("Angular Rate Error ω_BR vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("ω (rad/s)")
        plt.legend(['ω1', 'ω2', 'ω3'])
        plt.tight_layout()

        # Mission Modes
        plt.figure()
        plt.plot(time_history, mode_history, drawstyle='steps-post')
        plt.title("Vehicle Mode History vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Mode")
        plt.grid(True)
        plt.yticks(
            [INIT, SUN_MODE, NADIR_MODE, COMMS_MODE],
            ["INIT", "SUN_MODE", "NADIR_MODE", "COMMS_MODE"]
        )
        
        # Control and Disturbance Torques
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(time_history, control_history)
        plt.title("Control Torque over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Torque (N·m)")
        plt.legend(['τ1', 'τ2', 'τ3'])
        
        plt.subplot(2, 1, 2)
        plt.plot(time_history, torqueDisturbances_history)
        plt.title("Disturbance Torque over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Torque (N·m)")
        plt.legend(['τ1', 'τ2', 'τ3'])
        plt.tight_layout()

        plt.show()

    print("SIMULATION ENDED")
    
    if ANIM_3D:
        print("ANIMATING")
        ani = animate(
            inertialPosLMO_history, 
            inertialPosGMO_history, 
            measured_DCM_history, 
            attitudeError_DCM_history, 
            mode_history, 
            dt
        )
        print("ANIMATION ENDED")
    
    print("END")


if __name__ == "__main__":
    main()
