"""
Visualization and animation functions for spacecraft simulation.

This module provides 3D animation and plotting functions for visualizing
spacecraft attitude, orbit, and mission modes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have PyQt5 installed


# ============================================================
# 3D Geometry Helper Functions
# ============================================================

def rotate_points(points, R):
    """Rotate points by rotation matrix R."""
    return points @ R.T


def make_cube(center, size):
    """Create cube vertices and faces."""
    cx, cy, cz = center
    s = size / 2
    v = np.array([
        [cx-s, cy-s, cz-s],
        [cx+s, cy-s, cz-s],
        [cx+s, cy+s, cz-s],
        [cx-s, cy+s, cz-s],
        [cx-s, cy-s, cz+s],
        [cx+s, cy-s, cz+s],
        [cx+s, cy+s, cz+s],
        [cx-s, cy+s, cz+s],
    ])
    faces = [
        [v[0], v[1], v[2], v[3]],
        [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]],
        [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]],
        [v[0], v[3], v[7], v[4]],
    ]
    return v, faces


def make_panel(center, normal, width, height):
    """Create solar panel geometry."""
    normal = normal / np.linalg.norm(normal)
    tmp = np.array([1, 0, 0])
    if abs(np.dot(tmp, normal)) > 0.8:
        tmp = np.array([0, 1, 0])
    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    hw = width / 2
    hh = height / 2
    p1 = center + u*hw + v*hh
    p2 = center - u*hw + v*hh
    p3 = center - u*hw - v*hh
    p4 = center + u*hw - v*hh
    return [p1, p2, p3, p4]


def make_cylinder(center, axis, radius, length, n_points=20):
    """Create cylindrical antenna geometry."""
    axis = axis / np.linalg.norm(axis)
    tmp = np.array([0, 1, 0])
    if np.allclose(axis, tmp):
        tmp = np.array([0, 0, 1])
    u = np.cross(axis, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    v /= np.linalg.norm(v)
    theta = np.linspace(0, 2*np.pi, n_points)
    top = center + axis*length/2
    bottom = center - axis*length/2
    points_top = np.array([top + radius*np.cos(t)*u + radius*np.sin(t)*v for t in theta])
    points_bottom = np.array([bottom + radius*np.cos(t)*u + radius*np.sin(t)*v for t in theta])
    faces = []
    for i in range(n_points):
        j = (i+1) % n_points
        faces.append([points_bottom[i], points_bottom[j], points_top[j], points_top[i]])
    return faces


def get_key_frames(mode_history, DCM_error_history,
                   N_anim_max=120,
                   time_skip=20,
                   extra_mode_frames=50,
                   extra_rot_frames=1,
                   tolerance=0.01):
    """
    Select key frames for animation combining:
    - Sparse fast samples
    - Extra frames around mode transitions
    - Extra frames where DCM error indicates attitude rotation
    """
    from config import INIT
    
    N_total = len(mode_history)

    # Evenly spaced frames
    evenly_spaced = np.linspace(0, N_total-1, N_anim_max, dtype=int).tolist()

    # Mode-change frames + buffer
    mode_change_frames = [0]
    for i in range(1, N_total):
        if mode_history[i] != mode_history[i-1] and mode_history[i] != INIT:
            mode_change_frames.append(i)
            for k in range(-extra_mode_frames, extra_mode_frames+1):
                f = i + k
                if 0 <= f < N_total:
                    mode_change_frames.append(f)

    # Frames where DCM error indicates rotation
    DCM_trace = np.array([np.trace(D) for D in DCM_error_history])
    rotation_indices = np.where(np.abs(DCM_trace - 3) > tolerance)[0]

    rot_frames = []
    for f in rotation_indices:
        for k in range(-extra_rot_frames, extra_rot_frames+1):
            ff = f + k
            if 0 <= ff < N_total:
                rot_frames.append(ff)

    # Combine everything
    key_frames = sorted(set(evenly_spaced + mode_change_frames + rot_frames))

    # Optional thinning (but keep important frames untouched)
    thinned = []
    for idx, f in enumerate(key_frames):
        if (idx % time_skip == 0) or (f in mode_change_frames) or (f in rot_frames):
            thinned.append(f)

    return sorted(set(thinned))


# ============================================================
# Main 3D Animation Function
# ============================================================

def animate(inertialPosLMO_history,
            inertialPosGMO_history,
            measured_DCM_history,
            DCM_error_history,
            mode_history,
            dt=1,
            mars_radius=3390,
            mars_scale=0.25,
            sat_scale=4.0,
            gmo_scale=1.5):
    """
    Create 3D animation of spacecraft orbit and attitude.
    
    Parameters
    ----------
    inertialPosLMO_history : array-like
        LMO satellite position history
    inertialPosGMO_history : array-like
        GMO satellite position history
    measured_DCM_history : array-like
        Attitude DCM history
    DCM_error_history : array-like
        Attitude error DCM history
    mode_history : array-like
        Mission mode history
    dt : float
        Timestep
    mars_radius : float
        Mars radius for visualization
    mars_scale : float
        Scaling factor for Mars
    sat_scale : float
        Scaling factor for satellite
    gmo_scale : float
        Scaling factor for GMO satellite
        
    Returns
    -------
    ani : FuncAnimation
        Animation object
    """
    from config import INIT, SUN_MODE, NADIR_MODE, COMMS_MODE
    
    inertialPosLMO_history = np.asarray(inertialPosLMO_history, float)
    inertialPosGMO_history = np.asarray(inertialPosGMO_history, float)/2
    measured_DCM_history   = np.asarray(measured_DCM_history, float)
    mode_history           = np.asarray(mode_history)

    sun_dir = np.array([0, 1, 0])  # +Y direction

    # Mars sphere
    R = mars_radius * mars_scale
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    x_m = R*np.outer(np.cos(u), np.sin(v))
    y_m = R*np.outer(np.sin(u), np.sin(v))
    z_m = R*np.outer(np.ones_like(u), np.cos(v))

    # Simple shading according to sun
    normals = np.array([np.array([np.cos(uu)*np.sin(vv),
                                  np.sin(uu)*np.sin(vv),
                                  np.cos(vv)])
                        for uu in u for vv in v]).reshape(len(u), len(v), 3)
    intensity = np.clip(np.tensordot(normals, sun_dir, axes=([2], [0])), 0, 1)
    colors = plt.cm.Reds(intensity)

    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_m, y_m, z_m, facecolors=colors, linewidth=0, antialiased=False)

    # Orbits
    ax.plot(*inertialPosLMO_history.T, 'b--', linewidth=1)
    ax.plot(*inertialPosGMO_history.T, 'g--', linewidth=1)

    # Satellite scales
    cube_size_LMO = 400 * sat_scale
    cube_size_GMO = 350 * gmo_scale
    panel_width  = 1200 * sat_scale
    panel_height = 200  * sat_scale

    # LMO collections
    lmo_cube = Poly3DCollection([], facecolors='blue', edgecolors='k', zorder=20)
    panel1 = Poly3DCollection([], facecolors='cyan', alpha=0.85, edgecolors='k', zorder=10)
    panel2 = Poly3DCollection([], facecolors='cyan', alpha=0.85, edgecolors='k', zorder=10)
    lmo_antenna = Poly3DCollection([], facecolors='silver', edgecolors='k', zorder=30)
    ax.add_collection3d(lmo_cube)
    ax.add_collection3d(panel1)
    ax.add_collection3d(panel2)
    ax.add_collection3d(lmo_antenna)

    # GMO collection
    gmo_cube = Poly3DCollection([], facecolors='green', edgecolors='k', zorder=15)
    ax.add_collection3d(gmo_cube)

    # Axes placeholders
    body_x = ax.quiver(0, 0, 0, 1, 0, 0, length=0)
    body_y = ax.quiver(0, 0, 0, 0, 1, 0, length=0)
    body_z = ax.quiver(0, 0, 0, 0, 0, 1, length=0)

    # Sun direction
    sun_dir_vis = np.array([0, -1, 0])  # pointing toward Mars along -Y

    # Draw fixed sun rays far along +Y
    n_sun_rays = 5
    sun_lines = []
    sun_distance = R * 7  # far away along +Y
    for i in range(n_sun_rays):
        x_offset = (i - 2) * R * 0.3
        z_offset = 0
        start = np.array([x_offset, sun_distance, z_offset])
        end = start + sun_dir_vis * 0.1 * sun_distance  # toward Mars
        line = Line3DCollection([[start, end]], colors='yellow', linewidths=2)
        ax.add_collection3d(line)
        sun_lines.append(line)

    # Add SUN text
    ax.text(0, sun_distance + R*0.5, 0, "SUN", color="yellow", fontsize=14)

    # Communication line
    comm_line = Line3DCollection([], colors='black', linewidths=2, linestyle='--', zorder=35)
    ax.add_collection3d(comm_line)

    # Mode text
    mode_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=14)

    max_range = np.max(np.linalg.norm(inertialPosLMO_history, axis=1))*2.0
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("LMO Orbit Animation with Cylindrical Antenna and Sun Rays")

    # Precompute antenna in body frame
    antenna_length = cube_size_LMO * 1.0
    antenna_radius = cube_size_LMO * 0.08
    offset = np.array([-1*(cube_size_LMO/2 + antenna_length/2), 0, 0])
    antenna_faces_b = make_cylinder(offset, np.array([-1, 0, 0]), radius=antenna_radius, length=antenna_length)

    # Mode mapping
    mode_names = {INIT: "INIT_MODE", SUN_MODE: "SUN_MODE", NADIR_MODE: "NADIR_MODE", COMMS_MODE: "COMMS_MODE"}

    skip = 1
    arrow_linewidth = 1
    arrow_ratio = 0.1

    bx_text = ax.text(0, 0, 0, "", color='red')
    by_text = ax.text(0, 0, 0, "", color='green')
    bz_text = ax.text(0, 0, 0, "", color='blue')

    def update(i):
        nonlocal body_x, body_y, body_z
        if i % skip != 0:
            return []

        LMO = inertialPosLMO_history[i]
        GMO = inertialPosGMO_history[i]
        DCM = measured_DCM_history[i]

        # LMO Cube
        cube_vertices, cube_faces = make_cube(np.zeros(3), cube_size_LMO)
        rv = rotate_points(cube_vertices, DCM) + LMO
        faces_rot = []
        for face in cube_faces:
            idx = [np.where((cube_vertices==fv).all(axis=1))[0][0] for fv in face]
            faces_rot.append(rv[idx])
        lmo_cube.set_verts(faces_rot)

        # Panels
        half_c = cube_size_LMO / 2
        panel_offset = half_c * 1.2
        panel1_center_b = np.array([0, +panel_offset, 0])
        panel2_center_b = np.array([0, -panel_offset, 0])
        panel_normal_b = np.array([0, 0, 1])
        p1b = make_panel(panel1_center_b, panel_normal_b, panel_width, panel_height)
        p2b = make_panel(panel2_center_b, panel_normal_b, panel_width, panel_height)
        p1i = rotate_points(np.array(p1b), DCM) + LMO
        p2i = rotate_points(np.array(p2b), DCM) + LMO
        panel1.set_verts([p1i])
        panel2.set_verts([p2i])

        # Cylindrical antenna
        antenna_faces_i = [rotate_points(np.array(f), DCM) + LMO for f in antenna_faces_b]
        lmo_antenna.set_verts(antenna_faces_i)

        # GMO Cube
        gmo_vertices, gmo_faces = make_cube(GMO, cube_size_GMO)
        gmo_cube.set_verts(gmo_faces)

        # Axes arrows
        axis_len = 1200 * sat_scale
        tip_x = LMO + axis_len*DCM[:, 0]
        tip_y = LMO + axis_len*DCM[:, 1]
        tip_z = LMO + axis_len*DCM[:, 2]

        for arrow in [body_x, body_y, body_z]:
            arrow.remove()
        body_x = ax.quiver(*LMO, *(tip_x-LMO), color='red', length=1,
                           linewidth=arrow_linewidth, arrow_length_ratio=arrow_ratio)
        body_y = ax.quiver(*LMO, *(tip_y-LMO), color='green', length=1,
                           linewidth=arrow_linewidth, arrow_length_ratio=arrow_ratio)
        body_z = ax.quiver(*LMO, *(tip_z-LMO), color='blue', length=1,
                           linewidth=arrow_linewidth, arrow_length_ratio=arrow_ratio)

        bx_text.set_position((tip_x[0], tip_x[1]))
        bx_text.set_3d_properties(tip_x[2])
        bx_text.set_text("b1")

        by_text.set_position((tip_y[0], tip_y[1]))
        by_text.set_3d_properties(tip_y[2])
        by_text.set_text("b2")

        bz_text.set_position((tip_z[0], tip_z[1]))
        bz_text.set_3d_properties(tip_z[2])
        bz_text.set_text("b3")

        # COMMS_MODE line
        if mode_history[i] == COMMS_MODE:
            comm_line.set_segments([[LMO, GMO]])
        else:
            comm_line.set_segments([])

        # Mode text
        elapsed_time = i*dt
        mode_text.set_text(f"MODE: {mode_names.get(mode_history[i], 'UNKNOWN')} | TIME: {elapsed_time:.1f}s")

        return []

    key_frames = get_key_frames(mode_history, DCM_error_history)
    ani = FuncAnimation(fig, update, frames=key_frames, interval=100, blit=False)

    plt.show()
    return ani


def animate_orbit_and_attitude(inertialPosLMO_history, ref_attitudes, meas_attitudes, mode_history, time_history, interval=1):
    """
    Animate 3D orbit and coordinate frames of reference with attitude rotation.
    
    Parameters
    ----------
    inertialPosLMO_history : array-like
        Position history (Nx3 array)
    ref_attitudes : list of ndarrays
        Reference DCMs
    meas_attitudes : list of ndarrays
        Measured DCMs
    mode_history : list of ints
        Mode at each timestep
    time_history : list of floats
        Time at each timestep
    interval : int
        Milliseconds between frames
        
    Returns
    -------
    ani : FuncAnimation
        Animation object
    """
    from config import INIT, SUN_MODE, NADIR_MODE, COMMS_MODE
    
    # Define mode labels
    mode_labels = {INIT: "INIT", SUN_MODE: "SUN_MODE", NADIR_MODE: "NADIR_MODE", COMMS_MODE: "COMMS_MODE"}

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Orbit data
    pos_array = np.array(inertialPosLMO_history)
    ax.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], 'b', label='Orbit')
    ax.scatter([0], [0], [0], color='red', s=80, label='Mars')

    # Set axis limits
    max_range = np.max(np.abs(pos_array))
    margin = 1.2

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

    black_lines = []

    def update(frame):
        position = pos_array[frame]
        R_ref = ref_attitudes[frame]
        R_meas = meas_attitudes[frame]

        scale_factor = 1000*3

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
        black_lines.clear()

        # Direction to Mars
        direction_to_mars = np.array([0, 0, 0]) - position
        scaled_direction = 1.5 * direction_to_mars

        black_line = ax.plot([position[0], position[0] + scaled_direction[0]],
                             [position[1], position[1] + scaled_direction[1]],
                             [position[2], position[2] + scaled_direction[2]],
                             color='black', linewidth=0.5)
        black_lines.append(black_line[0])

        mode_text.set_text(f"Mode: {mode_labels.get(mode_history[frame], 'UNKNOWN')}")
        time_text.set_text(f"Time: {time_history[frame]:.1f} s")

        return ref_lines + meas_lines + [mode_text, time_text]

    ani = FuncAnimation(fig, update, frames=len(ref_attitudes), interval=interval, blit=False)
    plt.show()
    return ani


def animate_attitude(ref_attitudes, meas_attitudes, mode_history, time_history, interval=1):
    """
    Animate 3D coordinate frames of reference and measured attitude.
    
    Parameters
    ----------
    ref_attitudes : list of ndarrays
        Reference DCMs
    meas_attitudes : list of ndarrays
        Measured DCMs
    mode_history : list of ints
        Mode at each timestep
    time_history : list of floats
        Time at each timestep
    interval : int
        Milliseconds between frames
        
    Returns
    -------
    ani : FuncAnimation
        Animation object
    """
    from config import INIT, SUN_MODE, NADIR_MODE, COMMS_MODE
    
    mode_labels = {INIT: "INIT", SUN_MODE: "SUN_MODE", NADIR_MODE: "NADIR_MODE", COMMS_MODE: "COMMS_MODE"}
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Attitude Animation')

    ref_lines = [ax.plot([0, 0], [0, 0], [0, 0], color=c, alpha=0.3)[0] for c in ['r', 'g', 'b']]
    meas_lines = [ax.plot([0, 0], [0, 0], [0, 0], color=c, alpha=1.0, linewidth=2)[0] for c in ['r', 'g', 'b']]
    mode_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=14)
    time_text = ax.text2D(0.02, 0.90, "", transform=ax.transAxes, fontsize=12)
    
    def update(frame):
        R_ref = ref_attitudes[frame]
        R_meas = meas_attitudes[frame]
        
        origin = np.zeros(3)
        
        for i, line in enumerate(ref_lines):
            line.set_data([origin[0], R_ref[0, i]], [origin[1], R_ref[1, i]])
            line.set_3d_properties([origin[2], R_ref[2, i]])
        
        for i, line in enumerate(meas_lines):
            line.set_data([origin[0], R_meas[0, i]], [origin[1], R_meas[1, i]])
            line.set_3d_properties([origin[2], R_meas[2, i]])
        
        mode_text.set_text(f"Mode: {mode_labels.get(mode_history[frame], 'UNKNOWN')}")
        time_text.set_text(f"Time: {time_history[frame]:.1f} s")
        
        return ref_lines + meas_lines + [mode_text]
    
    ani = FuncAnimation(fig, update, frames=len(ref_attitudes), interval=interval, blit=False)
    plt.show()
    return ani
