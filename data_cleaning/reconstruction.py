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

# Get raw module angles (relative to defined ground plane). Using normal vector of rigid body, relative to each origin axis.
def get_module_angle(df: pd.DataFrame, module_name: str):
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

    # Normal vector = v1 × v2 for each time step (cross product along last axis)
    module_normal = np.cross(v1, v2)  # shape (n_samples, 3)

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

if __name__ == "__main__":
    df = limits.csv_to_df(limits.file_name, limits.file_dir)
    x_angle, y_angle, z_angle = get_module_angle(df, "FrontMod")
    marker_1, marker_2 = get_top_marker_names(df)
    dist = get_marker_distance(df, marker_1, marker_2)

    ax = plot_angles(df, x_angle, y_angle, z_angle)
    # ax = plot_marker_distance(df, dist)
    plot_robot_flat_times(get_robot_flat_times(df, dist), ax)
    limits.show_plot()
