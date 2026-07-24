import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import limits
import reconstruction as recon
import math

# Get movement mode
def get_movement_mode(df, front_mod_normal, back_mod_normal):
    """
    Determine movement mode based on the angle between front and back modules.
    
    Uses the dot product between module normal vectors to analyze how the angle
    between modules changes over time. A smaller dot product indicates a more 
    cat-arch (perpendicular) configuration.
    
    Parameters:
        df: DataFrame with time index
        front_mod_normal: array of front module normal vectors (n_samples, 3)
        back_mod_normal: array of back module normal vectors (n_samples, 3)
    
    Returns:
        dict with:
            - 'dot_product': time series of dot products between module normals
            - 'dot_product_deltas': time series of dot product changes
            - 'movement_mode': 'cat_arch' or 'arch' based on angle analysis
    """
    # Using the raw normal vectors to find how the angle between modules is changing (using raw to avoid vector sign change)
    # Dot product will always be positive since these are both always positive
    # The more cat-arch, the smaller (and possibly more negative) the dot product
    dot_prod = np.array([np.dot(front_mod_normal[i], back_mod_normal[i]) for i in range(len(front_mod_normal))])
    
    # Compute deltas (changes in dot product over time)
    dot_prod_deltas = np.diff(dot_prod)
    movement_modes = ['cat' if dot_prod_deltas[i] < 0 else 'cow' for i in range(len(dot_prod_deltas))]
    
    return movement_modes

# Returns only the times when movement mode changes
def sense_movement_mode(df, front_mod_normal, back_mod_normal, min_consecutive=10):
    """
    Detect movement mode changes (cat, cow, rest) using consecutive-count logic.
    
    Similar to sense_robot_pickup, this function requires a minimum number of 
    consecutive samples in the same mode before registering a mode change.
    
    Parameters:
        df: DataFrame with time index
        front_mod_normal: array of front module normal vectors (n_samples, 3)
        back_mod_normal: array of back module normal vectors (n_samples, 3)
        min_consecutive: minimum consecutive samples to confirm a mode change (default 30)
    
    Returns:
        DataFrame with columns:
            - 'start_time': timestamp when mode change occurred
            - 'mode': the new movement mode ('cat', 'cow', or 'rest')
    """
    # Get per-sample movement modes
    movement_modes = get_movement_mode(df, front_mod_normal, back_mod_normal)
    
    # Create a Series with the same index as df (shifted by 1 due to diff)
    mode_series = pd.Series(movement_modes, index=df.index[1:])
    
    # Start with 'rest' as the default mode
    start_times = []
    modes = []
    current_mode = 'rest'
    
    i = 0
    while i < len(mode_series):
        # Scan forward for a mode change
        if mode_series.iloc[i] != current_mode:
            # Count how many consecutive samples have the new mode
            j = i + 1
            while j < len(mode_series) and mode_series.iloc[j] == mode_series.iloc[i]:
                j += 1
            consecutive_count = j - i
            
            if consecutive_count >= min_consecutive:
                # Register the mode change
                new_mode = mode_series.iloc[i]
                start_time = float(mode_series.index[i])
                start_times.append(start_time)
                modes.append(new_mode)
                current_mode = new_mode
                i = j
            else:
                # Brief spike below min_consecutive - skip it
                i = j
        else:
            i += 1
    
    return pd.DataFrame({
        'start_time': start_times,
        'mode': modes
    })

# Plot movement modes (can be @ each time or only @ change times)
def plot_movement_modes(df, movement_modes, ax=None):
    t = df.index.to_series()

    # Support two input types for `movement_modes`:
    # - DataFrame with columns ['start_time', 'mode'] (from sense_movement_mode)
    # - list/iterable of per-sample modes aligned with df.index[1:] (from get_movement_mode)
    if isinstance(movement_modes, pd.DataFrame):
        # Use provided start times directly
        start_times = movement_modes['start_time'].values
        modes = movement_modes['mode'].values
        cat_t = [st for st, m in zip(start_times, modes) if m == 'cat']
        cow_t = [st for st, m in zip(start_times, modes) if m == 'cow']
    else:
        # movement_modes is a list aligned with df.index[1:]; prepend placeholder for first index
        movement_to_binary = [1 if mode == 'cat' else 0 for mode in movement_modes]
        movement_to_binary = [0] + movement_to_binary
        cat_t = [df.index[i] for i, val in enumerate(movement_to_binary) if val == 1]
        cow_t = [df.index[i] for i, val in enumerate(movement_to_binary) if val == 0]

    if ax is None:
        ax = plt.subplot()
    
    ax.set(xlabel='time (s)')

    cat_label_added = False
    cow_label_added = False

    for t in cat_t:
        ax.axvline(x=t, color='green', linestyle=':', label='Start crossing front obstacle' if not cat_label_added else None)
        cat_label_added = True

    for t in cow_t:
        ax.axvline(x=t, color='red', linestyle=':', label='Start crossing back obstacle' if not cow_label_added else None)
        cow_label_added = True

    # ax.plot(t, movement_to_binary, color="blue", label='Movement Modes (1 is cat and 0 is cow)')
    ax.set(xlabel='time (s)')
    ax.legend()
    ax.grid()

    return ax

# Plot com z progress
def plot_com_z(df: pd.DataFrame, com_z, ax=None):
    t = df.index.to_series()

    ax = plt.subplot() if ax is None else ax
    ax.plot(t, com_z, color="purple", linestyle='--', label='Robot Estimated CoM z (m)')

# Plot all pvc z's and com z over time
def plot_pvc_com_z(df: pd.DataFrame, pvc_z, com_z, ax=None):
    t = df.index.to_series()
    active_pvc = recon.get_active_pvc_names(df)
    pvc_active_colors = {
        'pvc1': 'red',
        'pvc2': 'yellow', 
        'pvc3' : 'green',
        'pvc4' : 'purple'
    }

    ax = plt.subplot() if ax is None else ax
    plot_com_z(df, com_z, ax)

    # TODO: Might want to plot these on a separate y axis to increase scale readibility

    inactive_label_added = False
    for i in range(1, 5):
        pvc_name = f"pvc{i}"

        # Plot the pvc z values for each obstacle
        ax.plot(
            t,
            pvc_z[:, i-1],
            linestyle=':',
            color="gray",
            alpha=0.25,
            label='inactive pvc' if not inactive_label_added else None,
        )
        inactive_label_added = True

        # Overlay active segments with solid linestyle, labeling only the first active segment
        active_label_added = False
        for sample_idx in range(len(df)):
            if sample_idx < len(active_pvc) and pvc_name in active_pvc[sample_idx]:
                ax.plot(
                    t.iloc[sample_idx:sample_idx+2],
                    pvc_z[sample_idx:sample_idx+2, i-1],
                    linestyle='-',
                    color=pvc_active_colors[pvc_name],
                    alpha=0.5,
                    label=(pvc_name + " Δz (active)") if not active_label_added else None,
                )
                active_label_added = True

    start_t, stop_t = limits.time_limits(df)
    ax.set(xlabel='time (s)')
    ax.set_ylabel(ylabel='z (m)')
    ax.set_xlim(start_t, stop_t)
    ax.legend()
    ax.grid()

    return ax

# Plot the robot com's y and z velocities over time (with the option to instead plot over z position)
def plot_y_z_vels(df: pd.DataFrame, com_y_vel, com_z_vel, com_z=None):
    t = df.index.to_series()
    x_axis = t if com_z is None else com_z

    ax = plt.subplot()
    ax.plot(x_axis, com_y_vel, color="blue", label='Robot CoM y-velocity (m)')
    ax.plot(x_axis, com_z_vel, color="orange", label='Robot CoM z-velocity (m)')

    start_t, stop_t = limits.time_limits(df)
    ax.set(xlabel='time (s)')
    ax.set_ylabel(ylabel='velocity (m/s)')
    ax.set_xlim(start_t, stop_t)
    ax.legend()
    ax.grid()

    return ax

if __name__ == "__main__":
    df = limits.csv_to_df(limits.file_name, limits.file_dir)
    marker_1, marker_2 = recon.get_top_marker_names(df)
    dist = recon.get_marker_distance(df, marker_1, marker_2)
    flat_times = recon.get_robot_flat_times(df, dist)

    # Corrected (true) normals
    true_normals = recon.get_module_normals(df, recon.get_empirical_normal_offsets(df, flat_times))
    true_front_bottom, true_back_bottom, _raw_top, _raw_bot = recon.get_top_marker_bottom_points(df, true_normals)

    # Raw (uncorrected) normals from marker body plane
    front_mod_name, back_mod_name = limits.get_module_names(df)
    raw_front_normals = recon.get_marker_body_normal(df, front_mod_name)
    raw_back_normals  = recon.get_marker_body_normal(df, back_mod_name)
    raw_module_normals = (raw_front_normals, raw_back_normals)
    raw_bottom_pts = recon.get_top_marker_bottom_points(df, raw_module_normals)

    # PVC coordinates
    pvc_z_displacements, pvc_initial_z = recon.get_pvc_z_displacements(df, ["pvc1", "pvc2", "pvc3", "pvc4"])
    # print(pvc_z_displacements.shape)
    pvc_z = pvc_z_displacements
    # print(pvc_z.shape)

    # CoM coordinates/velocities
    com_points = recon.get_com_points(df, true_front_bottom, true_back_bottom)
    com_z = com_points[:, 2]

    print(f"com_points shape: {com_points.shape}")
    # get_marker_velocity expects a list of position arrays, each (n_samples, 3)
    com_vel = np.array(recon.get_marker_velocity(df, marker_pos=[com_points]))
    smoothed_com_vel = np.array(recon.get_smoothed_marker_velocity(df, marker_pos=[com_points], window_size = 20))
    # com_vel shape is (1, 3, n_samples) - extract y and z velocity
    com_y_vel = smoothed_com_vel[0, :, 1]  # y velocity
    com_z_vel = smoothed_com_vel[0, :, 2]  # z velocity
    print(f"com_vel shape: {com_vel.shape}")
    print(f"com_y_vel shape: {com_y_vel.shape}")
    print(f"com_z_vel shape: {com_z_vel.shape}")

    # Finding movement modes
    movement_modes = get_movement_mode(df, raw_front_normals, raw_back_normals)
    movement_modes_start_t = sense_movement_mode(df, raw_front_normals, raw_back_normals)

    # Soft obstacle analysis
    # ax = plot_pvc_com_z(df, pvc_z, com_z)

    # Stiff obstacle analysis
    ax = plot_y_z_vels(df, com_y_vel, com_z_vel)

    # Plot z progress of CoM
    plot_com_z(df, com_z, ax)

    # Plot movement modes
    plot_movement_modes(df, movement_modes_start_t, ax)
                                  
    limits.show_plot()

    # Static plots (optional)
    # ax = recon.plot_angles(df, true_normals[0], true_normals[1])
    # recon.plot_robot_flat_times(flat_times, ax)
    # recon.plot_body_points(df, top_marker_front, top_marker_back, true_normals)
    # print(recon.get_empirical_normal_offsets(df, flat_times))
    # limits.show_plot()