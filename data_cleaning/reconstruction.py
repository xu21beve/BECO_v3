import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import limits
import math

top_marker_dist_lims = [105e-3, 150e-3] # m
mod_vectors_1_0 = {  # sorted left-right, x, y, z
    "front": [[1e-2*(5.5-1.675), 1e-2*(-3.5-0.75-0.65), 1e-2*(1.0+0.4+1.4)], 
              [1e-2*(-5.5+1.9), 1e-2*(-3.5-0.75-0.75), 1e-2*(1.0+0.4+0.95)]],
    "back":  [[1e-2*(-5.5+1.2), 1e-2*(-3.65-0.75-1.85), 1e-2*(-1.0-0.4-3.675)],
              [1e-2*(5.5-5.3), 1e-2*(-3.65-0.75-0.5), 1e-2*(-1.0-0.4-3.77)]]}
mod_vectors_static = {
    "front": [[1e-2*(-1)*(-5.65/2+3.675), 1e-2*(-1.05-2.7-2.5/2), 1e-2*(-1)*(1.0+2.6)],
              [1e-2*(-5.65/2+3.675), 1e-2*(-1.05-2.7-2.5/2), 1e-2*(-1)*(1.0+2.6)]],
    "back":  [[1e-2*(-3.9/2+3.675), 1e-2*(-1.05-2.7-2.5/2), 1e-2*(1.0+2.6)],
              [1e-2*(-1)*(-3.9/2+3.675), 1e-2*(-1.05-2.7-2.5/2), 1e-2*(1.0+2.6)]]}
top_marker_lengths = { # sorted as [front, back]
    "0.5": [3.5 + (2.54*1.5), 3.65 + (2.54*1.5)],
    "0.8": [3.5 + (2.54*1.5), 3.65 + (2.54*1.5)],
    "1.0": [3.5 + (2.54*1.5), 3.65 + (2.54*1.5)],
    "static": [1.05 + (2.54*1.5), 1.05 + (2.54*1.5)]
}
top_marker_z_distance = { # sorted as [front, back]
    "0.5": [(38.1-1.0)+2.6, 25.4+2.6+(38.1-1.0)],
    "0.8": [(38.1-1.0)+2.6, 25.4+2.6+(38.1-1.0)],
    "1.0": [(38.1-1.0)+2.6, 25.4+2.6+(38.1-1.0)],
    "static": [1.0+2.6, 25.4+2.6+1.0]
}

# Ground plane normal (pointing upward)
GROUND_NORMAL = np.array([0.0, 1.0, 0.0])

# ---------------------------------------------------------------------------
# Conversion helpers between normal vectors and angle offsets
# ---------------------------------------------------------------------------

# Convert angle offsets (x_angle, y_angle, z_angle) in degrees to unit normal vectors
def _angles_to_normals(angle_tuple):
    x_deg, y_deg, z_deg = angle_tuple
    x_rad = np.radians(x_deg)
    y_rad = np.radians(y_deg)
    z_rad = np.radians(z_deg)
    nx = np.cos(x_rad)
    ny = np.cos(y_rad)
    nz = np.cos(z_rad)
    return np.column_stack((nx, ny, nz))  # shape (n_samples, 3)

# Convert unit normal vectors to angle offsets (x_angle, y_angle, z_angle) in degrees
def _normals_to_angles(normals):
    normals = np.asarray(normals, dtype=float)
    if normals.ndim == 1:
        normals = normals[np.newaxis, :]
    # Normalize
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    unit = normals / norm
    # Clip to avoid numerical issues
    dot_x = np.clip(unit[:, 0], -1.0, 1.0)
    dot_y = np.clip(unit[:, 1], -1.0, 1.0)
    dot_z = np.clip(unit[:, 2], -1.0, 1.0)
    x_angle = np.degrees(np.arccos(dot_x))
    y_angle = np.degrees(np.arccos(dot_y))
    z_angle = np.degrees(np.arccos(dot_z))
    return x_angle, y_angle, z_angle

# Rotate vectors by the rotation that maps from_dir to to_dir
def _rotate_by_alignment(v, from_dir, to_dir):
    """Rotate vector(s) v by the rotation that maps from_dir to to_dir."""
    v = np.asarray(v, dtype=float)
    from_dir = np.asarray(from_dir, dtype=float)
    to_dir = np.asarray(to_dir, dtype=float)
    from_dir = from_dir / np.linalg.norm(from_dir)
    to_dir = to_dir / np.linalg.norm(to_dir)

    axis = np.cross(from_dir, to_dir)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-12:
        # Parallel or anti-parallel
        if np.dot(from_dir, to_dir) > 0:
            return v
        return -v

    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(from_dir, to_dir), -1.0, 1.0))
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Rodrigues' rotation formula
    v_rot = v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1 - cos_a)
    return v_rot

# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

# Plot module angles (from normals)
def plot_angles(df: pd.DataFrame, front_normals, back_normals):
    t = df.index
    fx, fy, fz = _normals_to_angles(front_normals)
    bx, by, bz = _normals_to_angles(back_normals)

    fig, ax = plt.subplots()
    ax.plot(t, fx, color="blue", label='Front x')
    ax.plot(t, fy, color="orange", label='Front y')
    ax.plot(t, fz, color="green", label='Front z')
    ax.plot(t, bx, color="blue", linestyle='--', label='Back x')
    ax.plot(t, by, color="orange", linestyle='--', label='Back y')
    ax.plot(t, bz, color="green", linestyle='--', label='Back z')
    ax.set(xlabel='time (s)')
    ax.set_ylabel(ylabel='angle from ground plane (deg)')
    ax.legend()
    ax.grid()

    return ax

# Plot module marker distance
def plot_marker_distance(df: pd.DataFrame, dist):
    t = df.index

    ax = plt.subplot()
    ax.plot(t, dist, color="blue")
    ax.axhline(y=(max(dist) + min(dist))/2.0)
    ax.set(xlabel='time (s)')
    ax.set_ylabel(ylabel='euclidean distance between markers (m)', color="blue")
    ax.tick_params(axis='y', labelcolor="blue")

    return ax

# Plot times when body should be flat (according to marker distance)
def plot_robot_flat_times(flat_times, ax=None):
    if ax is None:
        ax = plt.subplot()
    
    for t in flat_times:
        ax.axvline(x=t, color='gray', linestyle=':', alpha=0.5)
    
    return ax

# Plot body points over time, optionally drawing the reconstructed normals
def plot_body_points(df: pd.DataFrame, front_mod_points, back_mod_points, module_normals=None):
    t = df.index.to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(front_mod_points[:, 2], front_mod_points[:, 0], t, color="blue")
    ax.plot(back_mod_points[:, 2], back_mod_points[:, 0], t, color="orange")

    if module_normals is not None:
        front_normals, back_normals = module_normals
        # Ensure upward-pointing
        front_n = front_normals.copy()
        back_n = back_normals.copy()
        front_n[front_n[:, 1] < 0] *= -1
        back_n[back_n[:, 1] < 0]   *= -1

    ax.set_xlabel('z (m)')
    ax.set_ylabel('x (m)')
    ax.set_zlabel('t (s)')

    return ax

# ---------------------------------------------------------------------------
# Data access helpers
# ---------------------------------------------------------------------------

# Get distance in xyz between top markers
def get_marker_distance(df: pd.DataFrame, marker_1: str, marker_2: str):
    p1 = limits.get_marker_pos_xyz(df, marker_1)  # (n_samples, 3)
    p2 = limits.get_marker_pos_xyz(df, marker_2)

    dist = []
    for i in range(len(df.index)):
        dist.append(math.dist(p1[i], p2[i]))

    return dist

# Get mod names
def get_mod_names(df: pd.DataFrame):
    return limits.get_module_names(df)

# Checking for cases when the top marker is Marker2 rather than Marker1
def get_top_marker_names(df: pd.DataFrame):
    return limits.get_top_marker_names(df)

# Find timestamps where robot is flat, using average top marker distance
def get_robot_flat_times(df: pd.DataFrame, dist):
    mean_marker_dist = (min(dist) + max(dist))/2.0
    flat_times = []

    if mean_marker_dist < top_marker_dist_lims[0] or  mean_marker_dist > top_marker_dist_lims[1]:
        print(f"Top marker mean distance of {mean_marker_dist} m is outside of the reasonable bounds of {top_marker_dist_lims}. Using substitute distance of 14.5mm.")
        mean_marker_dist = 0.135

    for i in range(len(dist)):
        if abs(dist[i] - mean_marker_dist) <= 1e-3:
            flat_times.append(df.index[i])
    
    return flat_times

# ---------------------------------------------------------------------------
# Normal computation from marker positions
# ---------------------------------------------------------------------------

# Compute the unit normal vector of a plane defined by two edge vectors
# v_1 and v_2 can be single vectors (3,) or batches (n_samples, 3)
def get_plane_normal(v_1, v_2):
    v_1 = np.asarray(v_1, dtype=float)
    v_2 = np.asarray(v_2, dtype=float)

    if v_1.ndim == 1:
        v_1 = v_1[np.newaxis, :]
    if v_2.ndim == 1:
        v_2 = v_2[np.newaxis, :]

    # Normal vector = v1 × v2
    module_normal = np.cross(v_1, v_2)

    # Normalize to unit vectors
    norm = np.linalg.norm(module_normal, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    module_normal_unit = module_normal / norm

    return module_normal_unit  # shape (n_samples, 3)

# Get raw module normal vectors from the three markers of a rigid body
def get_marker_body_normal(df: pd.DataFrame, module_name: str):
    markers = [f"{module_name}:Marker1", f"{module_name}:Marker2", f"{module_name}:Marker3"]
    points = []
    for marker in markers:
        try:
            points.append(limits.get_marker_pos_xyz(df, marker))
        except KeyError:
            raise KeyError(f"Marker column '{marker}' not found in DataFrame.")

    p1, p2, p3 = points  # each is (n_samples, 3)

    v1 = p2 - p1
    v2 = p3 - p1

    return get_plane_normal(v1, v2)  # shape (n_samples, 3)

# ---------------------------------------------------------------------------
# Flatness and offset computation (all in normal-vector space)
# ---------------------------------------------------------------------------

# Get the robot normal when flat, using the two top markers and the other two front markers
def get_robot_normal_when_flat(df: pd.DataFrame, flat_times):
    if not flat_times:
        print("No flat times provided — returning None.")
        return None

    flat_t = flat_times[0]

    top_marker_1, top_marker_2 = get_top_marker_names(df)
    front_mod_name = top_marker_1.split(":")[0]

    front_top_marker = top_marker_1.split(":")[1]
    other_markers = [m for m in ["Marker1", "Marker2", "Marker3"] if m != front_top_marker]
    front_other_marker_1 = f"{front_mod_name}:{other_markers[0]}"
    front_other_marker_2 = f"{front_mod_name}:{other_markers[1]}"

    def _get_pos(marker):
        return np.array([
            limits.get_marker_pos(df, marker, "X").loc[flat_t],
            limits.get_marker_pos(df, marker, "Y").loc[flat_t],
            limits.get_marker_pos(df, marker, "Z").loc[flat_t],
        ], dtype=float)

    p_top_1   = _get_pos(top_marker_1)
    p_top_2   = _get_pos(top_marker_2)
    p_other_1 = _get_pos(front_other_marker_1)
    p_other_2 = _get_pos(front_other_marker_2)

    v_1 = p_top_2 - p_top_1
    v_2 = p_other_2 - p_other_1

    normal = get_plane_normal(v_1, v_2)
    return normal[0]  # single 3D vector

# Get the empirical normal offsets of both modules
# Returns the raw normal at the flat time (the "offset normal").
# The true normal at flat time should be GROUND_NORMAL = (0, 1, 0).
def get_empirical_normal_offsets(df: pd.DataFrame, flat_times):
    if not flat_times:
        return None, None

    flat_t = flat_times[0]

    try:
        flat_i = df.index.get_loc(flat_t)
    except KeyError:
        flat_i = np.where(df.index == flat_t)[0][0]

    front_mod_name, back_mod_name = get_mod_names(df)
    front_raw_normals = get_marker_body_normal(df, front_mod_name)
    back_raw_normals  = get_marker_body_normal(df, back_mod_name)

    front_offset = front_raw_normals[flat_i].copy()
    back_offset  = back_raw_normals[flat_i].copy()

    return front_offset, back_offset

# Apply normal offsets to raw normals to get corrected (true) normals
# The offset is the raw normal at flat time. We rotate each raw normal
# by the rotation that maps the offset normal to GROUND_NORMAL.
def get_module_normals(df: pd.DataFrame, normal_offsets):
    front_offset, back_offset = normal_offsets

    front_mod_name, back_mod_name = get_mod_names(df)
    front_raw = get_marker_body_normal(df, front_mod_name)
    back_raw  = get_marker_body_normal(df, back_mod_name)

    # Rotate each raw normal by the rotation that maps the offset to GROUND_NORMAL
    front_corrected = np.array([_rotate_by_alignment(n, front_offset, GROUND_NORMAL) for n in front_raw])
    back_corrected  = np.array([_rotate_by_alignment(n, back_offset, GROUND_NORMAL) for n in back_raw])

    return front_corrected, back_corrected

# ---------------------------------------------------------------------------
# Bottom point computation from normals
# ---------------------------------------------------------------------------

# Get top marker bottom points using normal vectors directly
# FIXME: Weird, because it doesn't work for the raw angles. Commented out offsets for now
def get_top_marker_bottom_points(df: pd.DataFrame, module_normals):
    front_normals, back_normals = module_normals
    top_marker_1, top_marker_2 = get_top_marker_names(df)

    print(f"Top marker names are {top_marker_1} and {top_marker_2}")

    front_mod_name = top_marker_1.split(":")[0]
    back_mod_name = top_marker_2.split(":")[0]

    wire_key = str(limits.wire_diam)
    if wire_key not in top_marker_lengths:
        print(f"Warning: wire_diam '{wire_key}' not in top_marker_lengths; skipping bottom point computation.")
        return None, None
    front_length, back_length = top_marker_lengths[wire_key]

    p_top_front = limits.get_marker_pos_xyz(df, top_marker_1)
    p_top_back  = limits.get_marker_pos_xyz(df, top_marker_2)

    # Ensure normals point upward
    front_n = front_normals.copy()
    back_n  = back_normals.copy()
    front_n[front_n[:, 1] < 0] *= -1
    back_n[back_n[:, 1] < 0]   *= -1

    front_bottom = p_top_front - front_length * front_n
    back_bottom  = p_top_back  - back_length  * back_n

    return front_bottom, back_bottom, p_top_front, p_top_back

# ---------------------------------------------------------------------------
# Marker position helpers
# ---------------------------------------------------------------------------

# Get all the robot marker positions as a list of arrays, one per marker, each shape (n_samples, 3)
def get_all_robot_markers_pos(df: pd.DataFrame):
    marker_pos = []
    front_mod_name, back_mod_name = get_mod_names(df)
    mod_names = [front_mod_name, back_mod_name]

    for mod_name in mod_names:
        for i in range(1, 4):
            marker_name = f"{mod_name}:Marker{i}"
            # subscript = ".1" if not (marker_name, "Position", "X") in df.columns else ""
            x = limits.get_marker_pos(df, marker_name, "X")
            y = limits.get_marker_pos(df, marker_name, "Y")
            z = limits.get_marker_pos(df, marker_name, "Z")
            marker_pos.append(np.column_stack((x, y, z)))

    return marker_pos  # list of 6 arrays: front M1, M2, M3, back M1, M2, M3

def get_pvc_marker_points(df: pd.DataFrame, pvc_name, marker_names=["Marker1", "Marker2", "Marker3"]):
    points = []

    for name in marker_names:
        x = limits.get_marker_pos(df, pvc_name + ":" + name, "X").to_numpy()
        y = limits.get_marker_pos(df, pvc_name + ":" + name, "Y").to_numpy()
        z = limits.get_marker_pos(df, pvc_name + ":" + name, "Z").to_numpy()
        points.append(np.column_stack((x, y, z)))
    
    return points

def get_all_pvc_marker_points(df: pd.DataFrame):
    pvc_names = ["pvc1", "pvc2", "pvc3", "pvc4"]
    points = []

    for name in pvc_names:
        points.extend(get_pvc_marker_points(df, name))
    
    return points

def get_pvc_velocity(df: pd.DataFrame, pvc_name):
    vel_vectors = []
    for name in pvc_name:
        t = df.index.to_series()
        # pose = limits.get_marker_pos_xyz(df, name)

        x_dot = limits.get_marker_pos(df, name ,'X').diff() / t.diff() # By only using the pvc_name, we are getting the rigid body's info
        y_dot = limits.get_marker_pos(df, name ,'Y').diff() / t.diff()
        z_dot = limits.get_marker_pos(df, name ,'Z').diff() / t.diff()

        vel_vectors.append([x_dot, y_dot, z_dot])
    
    return vel_vectors

# Identify active obstacles
# Could use velocity for spring-loaded tests, but since this doesn't work for static tests,
# perhaps another solution could be to locate the center point between the modules, and then
# see which two obstacles the module center lies between.
def get_active_pvc_names(df: pd.DataFrame):
    if limits.wire_diam == "static":
        print("No markers were placed on static obstacles. Exiting.")
        return []

    front_mod_name, back_mod_name = limits.get_module_names(df)
    if front_mod_name is None or back_mod_name is None:
        raise ValueError("Could not identify front and back module names in the dataframe.")

    # Only checking in the Z direction, since two pvc won't share the same Z value.
    mid_body_z = (
        limits.get_marker_pos(df, front_mod_name, "Z").to_numpy()
        + limits.get_marker_pos(df, back_mod_name, "Z").to_numpy()
    ) / 2.0

    # Build an (n_samples, 4) array of PVC Z positions and compare each sample to the
    # midpoint between the two modules.
    pvc_z = np.column_stack(
        [limits.get_marker_pos(df, f"pvc{i}", "Z").to_numpy() for i in range(1, 5)]
    )
    pvc_robot_delta_z = np.abs(pvc_z - mid_body_z[:, np.newaxis])

    # Get the two smallest delta-z values for each sample.
    closest_indices = np.argpartition(pvc_robot_delta_z, kth=1, axis=1)[:, :3]
    row_idx = np.arange(len(df))[:, np.newaxis]
    sort_order = np.argsort(pvc_robot_delta_z[row_idx, closest_indices], axis=1)
    closest_indices = np.take_along_axis(closest_indices, sort_order, axis=1)

    active_pairs = []
    for sample_idx, (i_1, i_2, i_3) in enumerate(closest_indices):
        i_1 = int(i_1) + 1
        i_2 = int(i_2) + 1
        i_3 = int(i_3) + 1

        # if abs(i_1 - i_2) != 1: # Commented out bc pvc being moved aren't always adjacent
        #     print(
        #         f"WARNING: sample {df.index[sample_idx]} active pvc are {i_1} and {i_2}, "
        #         "which are not adjacent."
        #     )

        active_pairs.append((f"pvc{i_1}", f"pvc{i_2}", f"pvc{i_3}"))

    return active_pairs

if __name__ == "__main__":
    df = limits.csv_to_df(limits.file_name, limits.file_dir)
    front_mod_name, _ = get_mod_names(df)
    front_normals = get_marker_body_normal(df, front_mod_name)
    marker_1, marker_2 = get_top_marker_names(df)
    dist = get_marker_distance(df, marker_1, marker_2)
    flat_times = get_robot_flat_times(df, dist)
    true_normals = get_module_normals(df, get_empirical_normal_offsets(df, flat_times))
    front_mod_points, back_mod_points, p_top_front, p_top_back = get_top_marker_bottom_points(df, true_normals)

    plot_body_points(df, front_mod_points, back_mod_points, true_normals)
    print(get_empirical_normal_offsets(df, flat_times))

    limits.show_plot()