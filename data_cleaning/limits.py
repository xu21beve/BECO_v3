import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# this is where we specify what trial we want to analyze
wire_diam = 0.5
pvc_spacing = 30
phi_fb = 0.7

file_dir = f'processed_data/{wire_diam}mm_Wire_Spring/'
file_name = f'{pvc_spacing}mm_{phi_fb}.csv'

def csv_to_df(filename, file_dir):
    df = pd.read_csv(file_dir + filename, header=[0,1,2], index_col=0)

    # Every remaining column is now purely numeric.
    df = df.apply(pd.to_numeric, errors="coerce")

    return df

# Plot vertical lines at the start and stop times
def plot_start_stop(df: pd.DataFrame):
    t = df.index
    z = df['FrontMod:Marker1', 'Position', 'Z.1']
    start_t, stop_t = time_limits(df)

    fig, ax = plt.subplots()
    ax.plot(t, z, color="blue")

    ax.set(xlabel='time (s)')
    ax.set_ylabel(ylabel='z (m)', color="blue")
    ax.tick_params(axis='y', labelcolor="blue")
    ax.axvline(x=start_t, color='green', linestyle=':')
    ax.axvline(x=stop_t, color='red', linestyle=':')
    ax.grid()

    return ax # For use in multi-y-axis plots

# Plot vertical lines at the pickup and setdown times
def plot_intervention(df: pd.DataFrame, start_t: float, stop_t: float, pickup_true_perimeter_false: bool, ax1=None):
    t = df.index
    y = df[("FrontMod", "Position", "Y")]
    
    events = sense_robot_pickup(df, start_t, stop_t) if pickup_true_perimeter_false else sense_robot_perimeter(df, start_t, stop_t)

    if not ax1:
        fig, ax = plt.subplots()
    else:
        ax = ax1.twinx()

    ax.plot(t, y, color="purple")

    ax.set(xlabel='time (s)')
    ax.set_ylabel(ylabel='y (m)', color="purple")
    ax.tick_params(axis='y', labelcolor="purple")
    ax.axvline(x=start_t, color='green', linestyle=':', label='start')
    ax.axvline(x=stop_t, color='red', linestyle=':', label='stop')
    ax.grid()

    # Plot each pickup and setdown event, numbered by their index
    for idx, row in events.iterrows():
        pt = row["start_time"]
        sd = row["stop_time"]

        # Pickup line (blue)
        ax.axvline(x=pt, color='blue', linestyle='--', alpha=0.7)
        ax.text(pt, ax.get_ylim()[1], f"P{idx}", rotation=90,
                verticalalignment='bottom', fontsize=8, color='blue')

        # Set-down line (orange) — only if not None
        if sd is not None:
            ax.axvline(x=sd, color='orange', linestyle='--', alpha=0.7)
            ax.text(sd, ax.get_ylim()[1], f"SD{idx}", rotation=90,
                    verticalalignment='bottom', fontsize=8, color='orange')

# Show the plot
def show_plot(timestamp_label=False):
    plt.title(f"wire_diam={wire_diam} | pvc_spacing={pvc_spacing} | phi_fb={phi_fb}")
    plt.savefig(f"limit_figs/wire_diam={wire_diam}-pvc_spacing={pvc_spacing}-phi_fb={phi_fb}.png" 
                if not timestamp_label else f"limit_figs/wire_diam={wire_diam}-pvc_spacing={pvc_spacing}-phi_fb={phi_fb}-timestamp={datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.png")
    plt.show()

# Plot the front module's and back module's path (along the x and z axes) as two separate lines 
# but on the same scale, and also plot horizontal and vertical lines of the perimeter as defined 
# by the PVC. 
def plot_mod_path(df: pd.DataFrame):
    x_f = df['FrontMod', 'Position', 'X']
    z_f = df['FrontMod', 'Position', 'Z']
    x_b = df['BackMod2', 'Position', 'X']
    z_b = df['BackMod2', 'Position', 'Z']
    start_t, stop_t = time_limits(df)

    fig, ax = plt.subplots()
    ax.plot(x_f, z_f, color="blue")
    ax.plot(x_b, z_b, color="orange")

    # Find PVC perimeter
    # Find the first time when pvc1 and pvc4 have non-empty position values
    pvc1_x = df[("pvc1", "Position", "X")]
    pvc4_x = df[("pvc4", "Position", "X")]

    # First index where both pvc1 and pvc4 have non-NaN position values
    valid_mask = pvc1_x.notna() & pvc4_x.notna()
    if not valid_mask.any():
        print("Note: No valid pvc1 and pvc4 positions found — no perimeter checking performed.")
        return None

    first_valid_idx = valid_mask.idxmax()

    # Collect x and z positions of all markers on pvc1 and pvc4 at the start
    pvc_markers = [
        "pvc1:Marker1", "pvc1:Marker2", "pvc1:Marker3",
        "pvc4:Marker1", "pvc4:Marker2", "pvc4:Marker3",
    ]

    x_positions = []
    z_positions = []

    for marker in pvc_markers:
        try:
            x_val = df.loc[first_valid_idx, (marker, "Position", "X")]
            z_val = df.loc[first_valid_idx, (marker, "Position", "Z")]
            if not (np.isnan(x_val) or np.isnan(z_val)):
                x_positions.append(x_val)
                z_positions.append(z_val)
        except (KeyError, ValueError, TypeError):
            pass

    if not x_positions or not z_positions:
        print("Note: Could not determine perimeter from PVC markers.")
        return None

    # Define the largest possible rectangle from the marker positions
    x_min, x_max = min(x_positions), max(x_positions)
    z_min, z_max = min(z_positions), max(z_positions)

    ax.set(xlabel='x (m)')
    ax.set_ylabel(ylabel='z (m)', color="blue")
    ax.tick_params(axis='y', labelcolor="blue")
    ax.axvline(x=x_min, color='green', linestyle=':')
    ax.axvline(x=x_max, color='green', linestyle=':')
    ax.axhline(y=z_min, color='green', linestyle=':')
    ax.axhline(y=z_max, color='green', linestyle=':')
    ax.grid()

    return ax # For use in multi-y-axis plots

# Check for times when the x and z values of both modules are outside of the perimeter defined
# by the starting values of the individual markers on pvc1 and pvc4 -- define start as when pvc1 and pvc4
# have non-empty position values. Define the largest possible rectangle. 
# Print a note that there is no perimeter checking occuring if wire_diam = static and then pass
# FIXME: Ignore pvc1 for 0.5, 20 mm, 0.61 phi_fb case
def sense_robot_perimeter(df: pd.DataFrame, start_t: float, stop_t: float):
    # Print a note if wire_diam is static (no perimeter checking)
    if isinstance(wire_diam, str) and wire_diam.lower() == "static":
        print("Note: Static test — no perimeter checking performed.")
        return None

    # Find the first time when pvc1 and pvc4 have non-empty position values
    pvc1_x = df[("pvc1", "Position", "X")]
    pvc4_x = df[("pvc4", "Position", "X")]

    # First index where both pvc1 and pvc4 have non-NaN position values
    valid_mask = pvc1_x.notna() & pvc4_x.notna()
    if not valid_mask.any():
        print("Note: No valid pvc1 and pvc4 positions found — no perimeter checking performed.")
        return None

    first_valid_idx = valid_mask.idxmax()

    # Collect x and z positions of all markers on pvc1 and pvc4 at the start
    pvc_markers = [
        "pvc1:Marker1", "pvc1:Marker2", "pvc1:Marker3",
        "pvc4:Marker1", "pvc4:Marker2", "pvc4:Marker3",
    ]

    x_positions = []
    z_positions = []

    for marker in pvc_markers:
        try:
            x_val = df.loc[first_valid_idx, (marker, "Position", "X")]
            z_val = df.loc[first_valid_idx, (marker, "Position", "Z")]
            if not (np.isnan(x_val) or np.isnan(z_val)):
                x_positions.append(x_val)
                z_positions.append(z_val)
        except (KeyError, ValueError, TypeError):
            pass

    if not x_positions or not z_positions:
        print("Note: Could not determine perimeter from PVC markers.")
        return None

    # Define the largest possible rectangle from the marker positions
    x_min, x_max = min(x_positions), max(x_positions)
    z_min, z_max = min(z_positions), max(z_positions)

    print(f"Perimeter bounds: x=[{x_min:.3f}, {x_max:.3f}], z=[{z_min:.3f}, {z_max:.3f}]")

    # Restrict to the time window [start_t, stop_t]
    window_mask = (df.index >= start_t) & (df.index <= stop_t)

    # Determine which back module column name is present
    back_mod_col = "BackMod" if ("BackMod", "Position", "X") in df.columns else \
                   "BackMod2" if ("BackMod2", "Position", "X") in df.columns else None

    # Check if FrontMod and BackMod go outside the perimeter
    front_x = df.loc[window_mask, ("FrontMod", "Position", "X")]
    front_z = df.loc[window_mask, ("FrontMod", "Position", "Z")]

    front_outside = (front_x < x_min) | (front_x > x_max) | (front_z < z_min) | (front_z > z_max)

    if back_mod_col is not None:
        back_x = df.loc[window_mask, (back_mod_col, "Position", "X")]
        back_z = df.loc[window_mask, (back_mod_col, "Position", "Z")]
        back_outside = (back_x < x_min) | (back_x > x_max) | (back_z < z_min) | (back_z > z_max)
    else:
        back_outside = pd.Series(False, index=front_outside.index)

    outside_mask = front_outside & back_outside # Both front and back mod must be out of bounds

    # Find contiguous runs of outside-perimeter events, which occurs for at least 0.24 seconds
    min_consecutive = 30
    group_id = (outside_mask != outside_mask.shift()).cumsum()
    groups = outside_mask.index.to_series().groupby(group_id)

    outside_runs = []
    for gid, idx_in_group in groups:
        if not outside_mask.loc[idx_in_group].iloc[0]:
            continue                                   # this is a "False" run, skip
        if len(idx_in_group) >= min_consecutive:
            outside_runs.append({
                "start_time": float(idx_in_group.iloc[0]),
                "stop_time": float(idx_in_group.iloc[-1]),
                "length": len(idx_in_group),
            })

    if outside_runs:
        print(f"Robot outside perimeter at {len(outside_runs)} interval(s):")
        for r in outside_runs:
            print(f"  {r['start_time']:.3f}s -> {r['stop_time']:.3f}s (len={r['length']})")
    else:
        print("Robot stayed within perimeter for entire trial.")

    return pd.DataFrame(outside_runs) if outside_runs else pd.DataFrame(columns=["start_time", "stop_time", "length"])

# Sense robot pickup during trial; run after start and stop time checks, to avoid detecting start and stop times as false positive pickups
def sense_robot_pickup(df: pd.DataFrame, start_t: float, stop_t: float):
    y_lim = 0.14

    # go through entire file, at first, try observing only one marker's z-axis (which is the interesting direction motion for mocap)
    y = df[("FrontMod", "Position", "Y")]

    # Restrict to the time window [start_t, stop_t]
    y_windowed = y.loc[start_t:stop_t]

    # find time stamps where the y-value moves more than 30% above the moving average, and then pair that time with a "set down" time, indicated by when the module's y-value falls below 110% of that same moving average
    above_y_lim = y_windowed > y_lim      # pickup threshold

    pickup_times = []
    set_down_times = []
    i = 0
    pickup_min_count = 30 # 0.25 sec/0.008 capture time ≈ 30 count

    while i < len(y_windowed):
        # Scan forward for a pickup event
        if above_y_lim.iloc[i]:
            # Count how many consecutive samples exceed the threshold
            j = i + 1
            while j < len(y_windowed) and above_y_lim.iloc[j]:
                j += 1
            consecutive_count = j - i

            if consecutive_count >= pickup_min_count:
                # Use the first timestamp of this consecutive run as pickup time
                pickup_time = float(y_windowed.index[i])
                print(f"Pickup time: {pickup_time} (consecutive: {consecutive_count})")
                # After pickup, scan forward for a set-down event
                k = j + 1
                while k < len(y_windowed) and above_y_lim.iloc[k]:
                    k += 1
                if k < len(y_windowed):
                    set_down_time = float(y_windowed.index[k])
                    pickup_times.append(pickup_time)
                    set_down_times.append(set_down_time)
                    i = k + 1
                else:
                    # No set-down found — fill None for this pickup
                    pickup_times.append(pickup_time)
                    set_down_times.append(None)
                    i += 1
            else:
                # Brief spike below min_count — skip it
                i = j + 1
        else:
            i += 1

    events = pd.DataFrame({
        "start_time": pickup_times,
        "stop_time": set_down_times,
    })
    return events

# Start and end time finder
def time_limits(df: pd.DataFrame):
    limit_slope = 2e-2  # any velocity below 1 cm/s is considered not moving
    min_consecutive = 3 # must occur for at least 0.024 seconds

    # go through entire file, at first, try observing only one marker's z-axis (which is the interesting direction motion for mocap)
    x = df.index.to_series()
    y = df[("FrontMod:Marker1", "Position", "Z.1")]
    slope_vectorized = y.diff() / x.diff()

    mask = slope_vectorized.to_numpy() > limit_slope          # boolean array, NaN -> False
    mask = (slope_vectorized.abs() > limit_slope).fillna(False)
 
    # Every time the mask flips (True->False or False->True), bump a
    # group id. Consecutive identical values share a group id.
    group_id = (mask != mask.shift()).cumsum()
 
    # For each group, keep only the True ones, then filter by run length.
    groups = slope_vectorized.index.to_series().groupby(group_id)
 
    runs = []
    for gid, idx_in_group in groups:
        if not mask.loc[idx_in_group].iloc[0]:
            continue                                   # this is a "False" run, skip
        if len(idx_in_group) >= min_consecutive:
            runs.append({
                "start_time": idx_in_group.iloc[0],
                "stop_time": idx_in_group.iloc[-1],
                "length": len(idx_in_group),
            })

    runs = pd.DataFrame(runs)

    first_idx = float(runs.iloc[0]["start_time"])
    last_idx = float(runs.iloc[-1]["stop_time"])
    return first_idx, last_idx

if __name__ == "__main__":
    df = csv_to_df(file_name, file_dir)
    start_t, stop_t = time_limits(df)
    events = sense_robot_pickup(df, start_t, stop_t)
    ax1 = plot_start_stop(df)
    plot_intervention(df, start_t, stop_t, False, ax1)
    # plot_mod_path(df)
    show_plot()
