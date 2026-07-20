import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import limits
import math

# formatted in the following hierarchy: wire diam, pvc spacing, phi_fb
front_marker2_is_top_cases = [
    [1.0, 30, 0.45], [1.0, 30, 0.86], 
    [1.0, 20, 0.2], [1.0, 20, 0.28],
    [0.8, 20, 0.9], [0.8, 20, 0.1], [0.8, 20, 0.35],
    [0.5, 20, 0.28], [0.5, 20, 0.27], [0.5, 20, 0.12]
]
back_marker2_is_top_cases = [
    ["static", 30, "all"]
]

top_marker_dist_lims = [105e-3, 150e-3] # m
mod_vectors_1_0 = {"front": [[1e-2*(5.5-1.675), 1e-2*(-3.5-0.75-0.65), 1e-2*(0.7+0.4+1.4)], 
                                  [1e-2*(-5.5+1.9), 1e-2*(-3.5-0.75-0.75), 1e-2*(0.7+0.4+0.95)]],
                    "back": [[1e-2*(-5.5+1.2), 1e-2*(-3.65-0.75-1.85), 1e-2*(-0.9-0.4-3.675)],
                                  [1e-2*(5.5-5.3), 1e-2*(-3.65-0.75-0.5), 1e-2*(-0.9-0.4-3.77)]]} # sorted left-right, x, y, z
mod_vectors_static = {"front": [[1e-2*(-1)*(-5.65/2+3.675), 1e-2*(-1.05-2.7-2.5/2), 1e-2*(-1)*(0.9+2.6)], # z= 0.35+0.7
                                  [1e-2*(-5.65/2+3.675), 1e-2*(-1.05-2.7-2.5/2), 1e-2*(-1)*(0.9+2.6)]],
                    "back": [[1e-2*(-3.9/2+3.675), 1e-2*(-1.05-2.7-2.5/2), 1e-2*(0.9+2.6)],
                                  [1e-2*(-1)*(-3.9/2+3.675), 1e-2*(-1.05-2.7-2.5/2), 1e-2*(0.9+2.6)]]}

# Get raw module angles (relative to defined ground plane). Using normal vector of rigid body, relative to each origin axis.
def get_marker_body_angle(df: pd.DataFrame, module_name: str):
    # Use all three module markers as points to define the plane, find the x, y, and z angles
    # relative to the ground plane, as defined by the normal vector <x, y, z> = <0, 1, 0>
    # and the origin (x, y, z) = (0, 0, 0)

    # Ground plane normal
    ground_normal = np.array([0.0, 1.0, 0.0])

    # Extract the three marker positions as (x, y, z) tuples
    markers = [f"{module_name}:Marker1", f"{module_name}:Marker2", f"{module_name}:Marker3"]
    points = []
    for marker in markers:
        try:
            x = df[(marker, "Position", "X")]
            y = df[(marker, "Position", "Y")]
            z = df[(marker, "Position", "Z")]
            points.append(np.column_stack((x, y, z)))
        except KeyError:
            raise KeyError(f"Marker column '{marker}' not found in DataFrame.")

    # Stack into a single array of shape (n_samples, 3, 3)
    p1, p2, p3 = points  # each is (n_samples, 3)

    # Compute two edge vectors for each time step: v1 = p2-p1, v2 = p3-p1
    v1 = p2 - p1
    v2 = p3 - p1

    return get_measured_angle_offsets(v1, v2)

# Plot module angles
def plot_angles(df: pd.DataFrame, x_angle, y_angle, z_angle):
    t = df.index

    fig, ax = plt.subplots()
    ax.plot(t, x_angle, color="blue")
    ax.plot(t, y_angle, color="orange")
    ax.plot(t, z_angle, color="green")
    # ax.legend()
    ax.set(xlabel='time (s)')
    ax.set_ylabel(ylabel='angle from ground plane (deg)', color="blue")
    ax.tick_params(axis='y', labelcolor="blue")

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

# Get distance in xyz between top markers (check if the trial, defined by wire_diam, pvc_spacing, or phi_fb, have the top markers switched)
def get_marker_distance(df: pd.DataFrame, marker_1: str, marker_2: str):
    marker1_subscript = ".1" if (marker_1, "Position", "X") in df.columns else ""
    marker2_subscript = ".1" if (marker_2, "Position", "X") in df.columns else ""

    x_1 = df[(marker_1, "Position", "X" + marker1_subscript)]
    y_1 = df[(marker_1, "Position", "Y" + marker1_subscript)]
    z_1 = df[(marker_1, "Position", "Z" + marker1_subscript)]

    x_2 = df[(marker_2, "Position", "X" + marker2_subscript)]
    y_2 = df[(marker_2, "Position", "Y" + marker2_subscript)]
    z_2 = df[(marker_2, "Position", "Z" + marker2_subscript)]

    dist = []

    for i in range(len(df.index)):
        dist.append(math.dist([x_1.iloc[i], y_1.iloc[i], z_1.iloc[i]], [x_2.iloc[i], y_2.iloc[i], z_2.iloc[i]]))

    return dist

# Get distance between two top markers, checking for cases when 
# the top marker is Marker2 rather than Marker1
def get_top_marker_names(df: pd.DataFrame):
    settings = [limits.wire_diam, limits.pvc_spacing, limits.phi_fb]

    front_mod_name = "FrontMod" if ("FrontMod", "Position", "X") in df.columns else \
                   "FrontBody" if ("FrontBody", "Position", "X") in df.columns else None

    back_mod_name = "BackMod" if ("BackMod", "Position", "X") in df.columns else \
                   "BackMod2" if ("BackMod2", "Position", "X") in df.columns else \
                    "BackBody" if ("BackBody", "Position", "X") in df.columns else None

    front_marker_name = "Marker1" if not settings in front_marker2_is_top_cases else "Marker2"
    back_marker_name = "Marker1" if not settings in back_marker2_is_top_cases else "Marker2"

    marker_1 = front_mod_name + ":" + front_marker_name
    marker_2 = back_mod_name + ":" + back_marker_name

    return marker_1, marker_2

# Find timestamps where robot is flat, using average top marker distance
def get_robot_flat_times(df: pd.DataFrame, dist):
    mean_marker_dist = (min(dist) + max(dist))/2.0
    flat_times = []

    # Break if the mean distance is outside of the feasible range
    if mean_marker_dist < top_marker_dist_lims[0] or  mean_marker_dist > top_marker_dist_lims[1]:
        print(f"Top marker mean distance of {mean_marker_dist} m is outside of the reasonable bounds of {top_marker_dist_lims}")
        return flat_times

    # Now, find values within dist which equal mean_marker_dist. Then, using their index, 
    # find the corresponding index value in df
    for i in range(len(dist)):
        if abs(dist[i] - mean_marker_dist) <= 1e-3: # If marker distance is within 1 mm of the mean_marker_dist
            flat_times.append(df.index[i])
    
    return flat_times

# Get measured plane angle offset from ground plane (only for static and 1.0 tests), defined by top marker point and normal vector
# basically get measured angle offset from horizontal, which we can readily subtract from raw plane offsets from horizontal
# v_1 and v_2 are the two vectors w/ tails @ top marker and heads @ bottom markers.
# They may be passed as either a single 3D vector (shape (3,)) or a batch of vectors (shape (n_samples, 3)).
def get_measured_angle_offsets(v_1, v_2):
    v_1 = np.asarray(v_1, dtype=float)
    v_2 = np.asarray(v_2, dtype=float)

    if v_1.ndim == 1:
        v_1 = v_1[np.newaxis, :]
    if v_2.ndim == 1:
        v_2 = v_2[np.newaxis, :]

    # Normal vector = v1 × v2 for each time step (cross product along last axis)
    module_normal = np.cross(v_1, v_2)  # shape (n_samples, 3)

    # Normalize to unit vectors
    norm = np.linalg.norm(module_normal, axis=1, keepdims=True)
    # Avoid division by zero
    norm[norm == 0] = 1.0
    module_normal_unit = module_normal / norm

    # Compute angles (in radians, converted to degrees) between module normal and each axis
    # Angle relative to ground normal (y-axis)
    dot_y = np.clip(module_normal_unit[:, 1], -1.0, 1.0)
    y_angle = np.degrees(np.arccos(dot_y))

    # Angle relative to x-axis
    dot_x = np.clip(module_normal_unit[:, 0], -1.0, 1.0)
    x_angle = np.degrees(np.arccos(dot_x))

    # Angle relative to z-axis
    dot_z = np.clip(module_normal_unit[:, 2], -1.0, 1.0)
    z_angle = np.degrees(np.arccos(dot_z))

    return x_angle, y_angle, z_angle

# Get the robot angle from when the robot is flat as measured by the top markers, approximately using the two markers on the front module
# for a rough measure of flatness along the x-axis
def get_robot_angle_when_flat(df: pd.DataFrame, flat_times):
    # Define the plane by finding the normal vector to the two vectors, 1) defined by the two top markers, and 
    # 2) defined by the other two markers on the front module. Use other other functions in the file as much as possible.
    # Use the first flat_time. Define v_1 as the vector between the two top markers and v_2 as the vector between the other two vectors, and then use get_measured_angle_offsets to output the plane, relative to the ground plane.

    if not flat_times:
        print("No flat times provided — returning None.")
        return None

    flat_t = flat_times[0]

    # Get the top marker names
    top_marker_1, top_marker_2 = get_top_marker_names(df)

    # Extract the front module name
    front_mod_name = top_marker_1.split(":")[0]

    # Determine which front markers are the top vs. the other two
    front_top_marker = top_marker_1.split(":")[1]  # e.g. "Marker1" or "Marker2"
    other_markers = [m for m in ["Marker1", "Marker2", "Marker3"] if m != front_top_marker]
    front_other_marker_1 = f"{front_mod_name}:{other_markers[0]}"
    front_other_marker_2 = f"{front_mod_name}:{other_markers[1]}"

    # Get positions at the first flat time
    def _get_pos(marker):
        subscript = ".1" if not (marker, "Position", "X") in df.columns else ""
        return np.array([
            df.loc[flat_t, (marker, "Position", "X" + subscript)],
            df.loc[flat_t, (marker, "Position", "Y" + subscript)],
            df.loc[flat_t, (marker, "Position", "Z" + subscript)],
        ], dtype=float)

    p_top_1   = _get_pos(top_marker_1)
    p_top_2   = _get_pos(top_marker_2)
    p_other_1 = _get_pos(front_other_marker_1)
    p_other_2 = _get_pos(front_other_marker_2)

    # v_1: vector between the two top markers
    v_1 = p_top_2 - p_top_1

    # v_2: vector between the other two markers on the front module
    v_2 = p_other_2 - p_other_1

    # Use get_measured_angle_offsets to compute plane angles
    x_angle, y_angle, z_angle = get_measured_angle_offsets(v_1, v_2)

    return float(x_angle[0]), float(y_angle[0]), float(z_angle[0])

# Get the empirical angle offsets of both modules from their marker planes
def get_empirical_angle_offsets(df: pd.DataFrame, flat_times):
    if not flat_times:
        return None, None

    flat_angles = get_robot_angle_when_flat(df, flat_times)
    flat_t = flat_times[0]

    # Resolve the flat timestamp to its position in the aligned angle arrays.
    try:
        flat_i = df.index.get_loc(flat_t)
    except KeyError:
        flat_i = np.where(df.index == flat_t)[0][0]

    module_angles_front = get_marker_body_angle(df, "FrontMod")
    module_angles_back = get_marker_body_angle(df, "BackMod" if "BackMod2" not in df.columns else "BackMod2")

    front_angle_offsets = tuple(
        angle[flat_i] - flat_angle
        for angle, flat_angle in zip(module_angles_front, flat_angles)
    )
    back_angle_offsets = tuple(
        angle[flat_i] - flat_angle
        for angle, flat_angle in zip(module_angles_back, flat_angles)
    )

    return front_angle_offsets, back_angle_offsets

# Get the true angle offsets of both modules from the ground plane
def get_module_angles(df: pd.DataFrame, angle_offsets):
    # Subtract angle offsets (which contains two 1x3 dimensional tuples of the front and back module angle offsets 
    # from the raw front and back module angles -- which each contain nx3 dimensional tuples.

    # Unpack offsets: each is a tuple (x_offset, y_offset, z_offset)
    front_offset, back_offset = angle_offsets
    front_ox, front_oy, front_oz = front_offset
    back_ox,  back_oy,  back_oz  = back_offset

    # Unpack raw angles: each is a tuple (x_angle, y_angle, z_angle) of arrays
    front_rx, front_ry, front_rz = get_marker_body_angle(df, "FrontMod")
    back_rx,  back_ry,  back_rz  = get_marker_body_angle(df, "BackMod") if "BackMod2" not in df.columns else get_marker_body_angle(df, "BackMod2") 

    # Subtract offsets element-wise (broadcasting over the time axis)
    front_x = front_rx - front_ox
    front_y = front_ry - front_oy
    front_z = front_rz - front_oz

    back_x = back_rx - back_ox
    back_y = back_ry - back_oy
    back_z = back_rz - back_oz

    return (front_x, front_y, front_z), (back_x, back_y, back_z)

# Get 
if __name__ == "__main__":
    df = limits.csv_to_df(limits.file_name, limits.file_dir)
    x_angle, y_angle, z_angle = get_marker_body_angle(df, "FrontMod")
    marker_1, marker_2 = get_top_marker_names(df)
    dist = get_marker_distance(df, marker_1, marker_2)
    flat_times = get_robot_flat_times(df, dist)
    true_angles = get_module_angles(df, get_empirical_angle_offsets(df, flat_times))

    ax = plot_angles(df, true_angles[0][0], true_angles[0][1], true_angles[0][2]) # frontmod
    # ax = plot_angles(df, true_angles[1][0], true_angles[1][1], true_angles[1][2]) # backmod
    # ax = plot_marker_distance(df, dist)
    plot_robot_flat_times(flat_times, ax)
    # print(f"Robot angle at first instance of flatness: {get_robot_angle_when_flat(df, flat_times)}")
    print(get_empirical_angle_offsets(df, flat_times))
    # print(get_measured_angle_offsets(mod_vectors_static["front"][0], mod_vectors_static["front"][1]))

    limits.show_plot()
