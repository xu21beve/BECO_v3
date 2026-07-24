import reconstruction as recon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import limits
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# create animation of the robot module planes moving in 3D, where the planes are defined by 
# but have the plane be defined by intersection with the corresponding top_marker_bottom_points output for that module,
# and the normal vector for the corresponding plane, derived from get_module_normals. Note that the data axes
# should correspond with the graph axes as follows: y corresponds to z, x corresponds to y, and z corresponds to x
# FIXME: Need to fix playback speed not increasing (only decreasing)
def animate_robot(df: pd.DataFrame, top_marker_bottom_points, module_normals, playback_speed=1.0, fig=None, ax=None):
    # FIXME: Swapped these two temporarily for the sake of comparison
    front_bottom, back_bottom, top_marker_front, top_marker_back, = top_marker_bottom_points  # each (n_samples, 3) in data coords
    front_normals, back_normals = module_normals              # each (n_samples, 3) unit normals

    # Ensure normals point upward
    front_n = front_normals.copy()
    back_n  = back_normals.copy()
    front_n[front_n[:, 1] < 0] *= -1
    back_n[back_n[:, 1] < 0]   *= -1

    # Plane half-size for visualization (in data units, ~size of the module)
    plane_half_size = 0.1

    # Coordinate transform: data (x, y, z) -> graph (z, x, y)
    def to_graph(p):
        return p[:, 2], p[:, 0], p[:, 1]

    # Build a square plane patch centered at bottom point, oriented by normal
    def make_plane_patch(bottom_pt, normal):
        # Find two orthogonal vectors in the plane
        if abs(normal[1]) < 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        else:
            ref = np.array([1.0, 0.0, 0.0])
        u = np.cross(normal, ref)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)

        # Four corners of the square plane
        corners_data = np.array([
            bottom_pt - plane_half_size * u - plane_half_size * v,
            bottom_pt + plane_half_size * u - plane_half_size * v,
            bottom_pt + plane_half_size * u + plane_half_size * v,
            bottom_pt - plane_half_size * u + plane_half_size * v,
        ])
        # Transform to graph coordinates
        corners_graph = np.column_stack(to_graph(corners_data))
        return corners_graph

    # Set up the figure (use provided or create new)
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    own_fig = fig

    # Determine axis limits from the data (in graph coordinates)
    all_pts = np.vstack([front_bottom, back_bottom])
    gx, gy, gz = to_graph(all_pts)
    margin = 0.1
    ax.set_xlim(gx.min() - margin, gx.max() + margin)
    ax.set_ylim(gy.min() - margin, gy.max() + margin)
    ax.set_zlim(gz.min() - margin, gz.max() + margin)
    ax.set_xlabel('Z (data) → X (graph)')
    ax.set_ylabel('X (data) → Y (graph)')
    ax.set_zlabel('Y (data) → Z (graph)')
    ax.set_title('Robot Module Planes Animation')

    # Initialize plane patches
    front_patch = Poly3DCollection([np.zeros((4, 3))], alpha=0.4, color='blue', label='Front Module')
    back_patch  = Poly3DCollection([np.zeros((4, 3))], alpha=0.4, color='orange', label='Back Module')
    ax.add_collection3d(front_patch)
    ax.add_collection3d(back_patch)

    # Initialize bottom point markers
    front_pt, = ax.plot([], [], [], 'bo', markersize=4)
    back_pt,  = ax.plot([], [], [], 'o', color='orange', markersize=4)

    # Initialize normal vectors as 3D arrows
    front_normal_vec = None
    back_normal_vec = None
    ax.legend()

    n_frames = len(df.index)

    def init():
        nonlocal front_normal_vec, back_normal_vec
        front_patch.set_verts([np.zeros((4, 3))])
        back_patch.set_verts([np.zeros((4, 3))])
        front_pt.set_data([], [])
        front_pt.set_3d_properties([])
        back_pt.set_data([], [])
        back_pt.set_3d_properties([])
        if front_normal_vec is not None:
            front_normal_vec.set_segments([])
        if back_normal_vec is not None:
            back_normal_vec.set_segments([])
        return front_patch, back_patch, front_pt, back_pt, front_normal_vec, back_normal_vec

    def update(frame):
        nonlocal front_normal_vec, back_normal_vec
        # Front module plane
        f_corners = make_plane_patch(front_bottom[frame], front_n[frame])
        front_patch.set_verts([f_corners])

        # Back module plane
        b_corners = make_plane_patch(back_bottom[frame], back_n[frame])
        back_patch.set_verts([b_corners])

        # Bottom points
        fg = to_graph(front_bottom[frame:frame+1])
        bg = to_graph(back_bottom[frame:frame+1])
        front_pt.set_data(fg[0], fg[1])
        front_pt.set_3d_properties(fg[2])
        back_pt.set_data(bg[0], bg[1])
        back_pt.set_3d_properties(bg[2])

        # Normal vectors as 3D arrows from the bottom points
        front_normal = front_n[frame]
        back_normal = back_n[frame]
        front_origin = front_bottom[frame]
        back_origin = back_bottom[frame]

        front_graph_origin = to_graph(front_origin[np.newaxis, :])
        back_graph_origin = to_graph(back_origin[np.newaxis, :])
        front_graph_origin = (front_graph_origin[0][0], front_graph_origin[1][0], front_graph_origin[2][0])
        back_graph_origin = (back_graph_origin[0][0], back_graph_origin[1][0], back_graph_origin[2][0])

        if front_normal_vec is not None:
            front_normal_vec.remove()
        if back_normal_vec is not None:
            back_normal_vec.remove()

        front_normal_vec = ax.quiver(
            front_graph_origin[0], front_graph_origin[1], front_graph_origin[2],
            front_normal[2], front_normal[0], front_normal[1],
            color='blue', alpha=0.5, length=2
        )
        back_normal_vec = ax.quiver(
            back_graph_origin[0], back_graph_origin[1], back_graph_origin[2],
            back_normal[2], back_normal[0], back_normal[1],
            color='orange', alpha=0.5, length=2
        )

        ax.set_title(f'Robot Module Planes — t = {df.index[frame]:.3f}s')
        return front_patch, back_patch, front_pt, back_pt, front_normal_vec, back_normal_vec

    # Adjust interval based on playback speed: lower interval = faster playback
    base_interval = 50  # ms at 1x speed
    interval = int(base_interval / playback_speed)

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        interval=interval, blit=False)

    plt.tight_layout()
    if ax is None:
        plt.show()
    return ani

# Animate all robot markers in 3D
def animate_robot_markers(df: pd.DataFrame, playback_speed=1.0, fig=None, ax=None):
    marker_pos_list = recon.get_all_robot_markers_pos(df)  # list of 6 arrays, each (n_samples, 3)
    front_colors = ['blue', 'cyan', 'deepskyblue']
    back_colors  = ['orange', 'gold', 'darkorange']
    labels = ['Front M1', 'Front M2', 'Front M3', 'Back M1', 'Back M2', 'Back M3']

    # Coordinate transform: data (x, y, z) -> graph (z, x, y)
    def to_graph(p):
        return p[:, 2], p[:, 0], p[:, 1]

    # Set up the figure (use provided or create new)
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    own_fig = fig

    # Determine axis limits from all markers
    all_pts = np.vstack(marker_pos_list)
    gx, gy, gz = to_graph(all_pts)
    margin = 0.1
    ax.set_xlim(gx.min() - margin, gx.max() + margin)
    ax.set_ylim(gy.min() - margin, gy.max() + margin)
    ax.set_zlim(gz.min() - margin, gz.max() + margin)
    ax.set_xlabel('Z (data) → X (graph)')
    ax.set_ylabel('X (data) → Y (graph)')
    ax.set_zlabel('Y (data) → Z (graph)')
    ax.set_title('All Robot Markers Animation')

    # Initialize scatter plot lines for each marker
    lines = []
    for i in range(6):
        color = front_colors[i] if i < 3 else back_colors[i - 3]
        ln, = ax.plot([], [], [], 'o', color=color, markersize=4, label=labels[i])
        lines.append(ln)
    ax.legend()

    n_frames = len(df.index)

    def init():
        for ln in lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        return lines

    def update(frame):
        for i, ln in enumerate(lines):
            gp = to_graph(marker_pos_list[i][frame:frame+1])
            ln.set_data(gp[0], gp[1])
            ln.set_3d_properties(gp[2])
        ax.set_title(f'All Robot Markers — t = {df.index[frame]:.3f}s')
        return lines

    base_interval = 50  # ms at 1x speed
    interval = int(base_interval / playback_speed)

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        interval=interval, blit=False)

    plt.tight_layout()
    if ax is None:
        plt.show()
    return ani

# Animate pvc markers and their velocity vectors
def animate_pvc_markers(df: pd.DataFrame, playback_speed=1.0, fig=None, ax=None):
    pvc_names = ["pvc1", "pvc2", "pvc3", "pvc4"]
    marker_pos_list = recon.get_all_pvc_marker_points(df)  # list of 12 arrays, each (n_samples, 3)
    pvc_colors = ['tomato', 'darkkhaki', 'slateblue', 'darkorchid']

    # Build one body position per PVC as the mean of its three marker positions.
    pvc_body_positions = []
    for i in range(len(pvc_names)):
        marker_group = marker_pos_list[i * 3:(i + 1) * 3]
        pvc_body_positions.append(np.mean(np.stack(marker_group, axis=0), axis=0))

    # Active PVC names for each sample.
    active_pvc_names = recon.get_active_pvc_names(df)

    # Velocity vectors for each PVC body.
    pvc_velocity_arrays = []
    for vel_components in recon.get_pvc_velocity(df, pvc_names):
        x_dot, y_dot, z_dot = vel_components
        vel_array = np.column_stack((
            x_dot.to_numpy(dtype=float),
            y_dot.to_numpy(dtype=float),
            z_dot.to_numpy(dtype=float),
        ))
        vel_array = np.nan_to_num(vel_array, nan=0.0, posinf=0.0, neginf=0.0)
        pvc_velocity_arrays.append(vel_array)

    # Coordinate transform: data (x, y, z) -> graph (z, x, y)
    def to_graph(p):
        return p[:, 2], p[:, 0], p[:, 1]

    # Set up the figure (use provided or create new)
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    own_fig = fig

    # Determine axis limits from all markers
    all_pts = np.vstack(marker_pos_list)
    gx, gy, gz = to_graph(all_pts)
    margin = 0.1
    ax.set_xlim(gx.min() - margin, gx.max() + margin)
    ax.set_ylim(gy.min() - margin, gy.max() + margin)
    ax.set_zlim(gz.min() - margin, gz.max() + margin)
    ax.set_xlabel('Z (data) → X (graph)')
    ax.set_ylabel('X (data) → Y (graph)')
    ax.set_zlabel('Y (data) → Z (graph)')
    ax.set_title('All PVC Markers Animation')

    # Initialize scatter plot lines for each marker and body point.
    marker_lines = []
    body_lines = []
    body_labels = []
    for i in range(len(pvc_body_positions)):
        color = pvc_colors[i % len(pvc_colors)]
        for marker_idx in range(3):
            ln, = ax.plot([], [], [], 'o', color=color, markersize=3, alpha=0.7)
            marker_lines.append(ln)
        body_ln, = ax.plot([], [], [], 'o', color=color, markersize=6, markerfacecolor='white', markeredgewidth=1.5)
        body_lines.append(body_ln)
        body_label = ax.text(0, 0, 0, '', fontsize=8, color=color)
        body_labels.append(body_label)
    ax.legend()

    velocity_vecs = []
    active_text_box = None
    n_frames = len(df.index)

    def init():
        nonlocal active_text_box
        for ln in marker_lines + body_lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        for label in body_labels:
            label.set_text('')
        for vec in velocity_vecs:
            vec.remove()
        velocity_vecs[:] = []
        if active_text_box is not None:
            active_text_box.remove()
            active_text_box = None
        return marker_lines + body_lines

    def update(frame):
        nonlocal active_text_box
        for old_vec in velocity_vecs:
            old_vec.remove()
        velocity_vecs[:] = []
        if active_text_box is not None:
            active_text_box.remove()
            active_text_box = None

        line_idx = 0
        for pvc_idx, color in enumerate(pvc_colors):
            marker_group = marker_pos_list[pvc_idx * 3:(pvc_idx + 1) * 3]
            for marker_offset in range(3):
                gp = to_graph(marker_group[marker_offset][frame:frame + 1])
                ln = marker_lines[line_idx]
                ln.set_data(gp[0], gp[1])
                ln.set_3d_properties(gp[2])
                ln.set_color(color)
                line_idx += 1

            body_gp = to_graph(pvc_body_positions[pvc_idx][frame:frame + 1])
            body_ln = body_lines[pvc_idx]
            body_ln.set_data(body_gp[0], body_gp[1])
            body_ln.set_3d_properties(body_gp[2])
            body_ln.set_color(color)

            body_labels[pvc_idx].set_position((body_gp[0][0], body_gp[1][0]))
            body_labels[pvc_idx].set_3d_properties(body_gp[2][0])
            body_labels[pvc_idx].set_text(pvc_names[pvc_idx])
            body_labels[pvc_idx].set_color(color)

            velocity = pvc_velocity_arrays[pvc_idx][frame]
            if np.linalg.norm(velocity) > 0:
                arrow_length = np.clip(np.linalg.norm(velocity) * 5.0, 0.01, 0.2)
                vec = ax.quiver(
                    body_gp[0][0], body_gp[1][0], body_gp[2][0],
                    velocity[2], velocity[0], velocity[1],
                    color=color,
                    alpha=0.7,
                    length=arrow_length,
                )
                velocity_vecs.append(vec)

        frame_timestamp = df.index[frame]
        active_names = active_pvc_names[frame] if frame < len(active_pvc_names) else []

        if active_names:
            label_parts = []
            for name in active_names:
                pvc_index = int(name[-1]) - 1
                color = pvc_colors[pvc_index % len(pvc_colors)]
                label_parts.append(f'{name} ')
            label_text = f'Active PVCs at t={frame_timestamp:.3f}s: ' + ', '.join(active_names)
        else:
            label_text = f'Active PVCs at t={frame_timestamp:.3f}s: none'

        active_text_box = ax.text(
            0.02, 0.95, 0.15,
            s=label_text,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            zorder=100,
            color='black',
        )

        for name in active_names:
            pvc_index = int(name[-1]) - 1
            color = pvc_colors[pvc_index % len(pvc_colors)]
            active_text_box.set_color(color)

        ax.set_title(f'All PVC Markers — t = {frame_timestamp:.3f}s')
        return marker_lines + body_lines + velocity_vecs + body_labels + [active_text_box]

    base_interval = 50  # ms at 1x speed
    interval = int(base_interval / playback_speed)

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        interval=interval, blit=False)

    plt.tight_layout()
    if ax is None:
        plt.show()
    return ani

# Animate both planes and markers together on the same plot.
# If raw_module_normals and raw_bottom_points are provided, raw (uncorrected) planes
# are also drawn as wireframes for comparison.
def animate_combined(df: pd.DataFrame, top_marker_bottom_points, module_normals,
                     playback_speed=1.0, raw_module_normals=None, raw_bottom_points=None):
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # --- Data preparation ---
    front_bottom, back_bottom, top_marker_front, top_marker_back, = top_marker_bottom_points
    front_normals, back_normals = module_normals
    marker_pos_list = recon.get_all_robot_markers_pos(df)
    robot_com_pos = recon.get_com_points(df, front_bottom, back_bottom)

    # Ensure corrected normals point upward
    front_n = front_normals.copy()
    back_n  = back_normals.copy()
    front_n[front_n[:, 1] < 0] *= -1
    back_n[back_n[:, 1] < 0]   *= -1

    # Raw (uncorrected) normals if provided
    has_raw = raw_module_normals is not None and raw_bottom_points is not None
    if has_raw:
        raw_front_normals, raw_back_normals = raw_module_normals
        raw_front_bottom, raw_back_bottom, raw_top_front, raw_top_back = raw_bottom_points

        # Ensure raw normals point upward
        raw_front_n = raw_front_normals.copy()
        raw_back_n  = raw_back_normals.copy()
        raw_front_n[raw_front_n[:, 1] < 0] *= -1
        raw_back_n[raw_back_n[:, 1] < 0]   *= -1

    plane_half_size = 0.1

    # Coordinate transform
    def to_graph(p):
        return p[:, 2], p[:, 0], p[:, 1]

    # Plane patch builder
    def make_plane_patch(bottom_pt, normal):
        if abs(normal[1]) < 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        else:
            ref = np.array([1.0, 0.0, 0.0])
        u = np.cross(normal, ref)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        corners_data = np.array([
            bottom_pt - plane_half_size * u - plane_half_size * v,
            bottom_pt + plane_half_size * u - plane_half_size * v,
            bottom_pt + plane_half_size * u + plane_half_size * v,
            bottom_pt - plane_half_size * u + plane_half_size * v,
        ])
        return np.column_stack(to_graph(corners_data))

    # --- Figure setup ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    plot_pts = [front_bottom, back_bottom] + marker_pos_list + robot_com_pos
    if has_raw:
        plot_pts += [raw_front_bottom, raw_back_bottom]
    all_pts = np.vstack(plot_pts)
    gx, gy, gz = to_graph(all_pts)
    margin = 0.1
    ax.set_xlim(gx.min() - margin, gx.max() + margin)
    ax.set_ylim(gy.min() - margin, gy.max() + margin)
    ax.set_zlim(gz.min() - margin, gz.max() + margin)
    ax.set_xlabel('Z (data) → X (graph)')
    ax.set_ylabel('X (data) → Y (graph)')
    ax.set_zlabel('Y (data) → Z (graph)')
    ax.set_title('Robot Animation — Planes + Markers')

    # --- Corrected (true) plane patches ---
    front_patch = Poly3DCollection([np.zeros((4, 3))], alpha=0.3, color='blue', label='Front (corrected)')
    back_patch  = Poly3DCollection([np.zeros((4, 3))], alpha=0.3, color='orange', label='Back (corrected)')
    ax.add_collection3d(front_patch)
    ax.add_collection3d(back_patch)

    # --- Raw (uncorrected) plane wireframes ---
    if has_raw:
        front_raw_patch = Poly3DCollection([np.zeros((4, 3))], alpha=0.15, color='red',
                                           label='Front (raw)', linestyle='--', linewidth=1,
                                           facecolors='none')
        back_raw_patch  = Poly3DCollection([np.zeros((4, 3))], alpha=0.15, color='green',
                                           label='Back (raw)', linestyle='--', linewidth=1,
                                           facecolors='none')
        ax.add_collection3d(front_raw_patch)
        ax.add_collection3d(back_raw_patch)
    else:
        front_raw_patch = None
        back_raw_patch = None

    # --- Corrected bottom points ---
    front_pt, = ax.plot([], [], [], 'bo', markersize=4)
    back_pt,  = ax.plot([], [], [], 'o', color='orange', markersize=4)
    robot_com_pt, = ax.plot([], [], [], 'o', color='black', markersize=5, label='COM')

    # --- Raw bottom points ---
    if has_raw:
        front_raw_pt, = ax.plot([], [], [], 'rx', markersize=4, label='Front (raw pt)')
        back_raw_pt,  = ax.plot([], [], [], 'gx', markersize=4, label='Back (raw pt)')
    else:
        front_raw_pt = None
        back_raw_pt = None

    # --- Normal vectors ---
    front_normal_vec = None
    back_normal_vec = None

    # --- Marker points ---
    front_colors = ['blue', 'cyan', 'deepskyblue']
    back_colors  = ['orange', 'gold', 'darkorange']
    labels = ['Front M1', 'Front M2', 'Front M3', 'Back M1', 'Back M2', 'Back M3']
    marker_lines = []
    for i in range(6):
        color = front_colors[i] if i < 3 else back_colors[i - 3]
        ln, = ax.plot([], [], [], 'o', color=color, markersize=4, label=labels[i])
        marker_lines.append(ln)
    ax.legend()

    n_frames = len(df.index)

    def init():
        nonlocal front_normal_vec, back_normal_vec
        front_patch.set_verts([np.zeros((4, 3))])
        back_patch.set_verts([np.zeros((4, 3))])
        if has_raw:
            front_raw_patch.set_verts([np.zeros((4, 3))])
            back_raw_patch.set_verts([np.zeros((4, 3))])
        front_pt.set_data([], []); front_pt.set_3d_properties([])
        back_pt.set_data([], []);  back_pt.set_3d_properties([])
        robot_com_pt.set_data([], []); robot_com_pt.set_3d_properties([])
        if has_raw:
            front_raw_pt.set_data([], []); front_raw_pt.set_3d_properties([])
            back_raw_pt.set_data([], []);  back_raw_pt.set_3d_properties([])
        for ln in marker_lines:
            ln.set_data([], []); ln.set_3d_properties([])
        if front_normal_vec is not None:
            front_normal_vec.set_segments([])
        if back_normal_vec is not None:
            back_normal_vec.set_segments([])
        ret = [front_patch, back_patch, front_pt, back_pt, robot_com_pt,
               front_normal_vec, back_normal_vec, *marker_lines]
        if has_raw:
            ret += [front_raw_patch, back_raw_patch, front_raw_pt, back_raw_pt]
        return ret

    def update(frame):
        nonlocal front_normal_vec, back_normal_vec

        # Corrected planes
        f_corners = make_plane_patch(front_bottom[frame], front_n[frame])
        front_patch.set_verts([f_corners])
        b_corners = make_plane_patch(back_bottom[frame], back_n[frame])
        back_patch.set_verts([b_corners])

        # Raw planes (wireframe)
        if has_raw:
            rf_corners = make_plane_patch(raw_front_bottom[frame], raw_front_n[frame])
            front_raw_patch.set_verts([rf_corners])
            rb_corners = make_plane_patch(raw_back_bottom[frame], raw_back_n[frame])
            back_raw_patch.set_verts([rb_corners])

        # Bottom points
        fg = to_graph(front_bottom[frame:frame+1])
        bg = to_graph(back_bottom[frame:frame+1])
        front_pt.set_data(fg[0], fg[1]); front_pt.set_3d_properties(fg[2])
        back_pt.set_data(bg[0], bg[1]);  back_pt.set_3d_properties(bg[2])

        com_gp = to_graph(robot_com_pos[0][frame:frame+1])
        robot_com_pt.set_data(com_gp[0], com_gp[1])
        robot_com_pt.set_3d_properties(com_gp[2])

        # Normals
        front_normal = front_n[frame]
        back_normal = back_n[frame]
        front_origin = front_bottom[frame]
        back_origin = back_bottom[frame]
        front_go = to_graph(front_origin[np.newaxis, :])
        back_go = to_graph(back_origin[np.newaxis, :])
        front_go = (front_go[0][0], front_go[1][0], front_go[2][0])
        back_go  = (back_go[0][0],  back_go[1][0],  back_go[2][0])

        if front_normal_vec is not None:
            front_normal_vec.remove()
        if back_normal_vec is not None:
            back_normal_vec.remove()
        front_normal_vec = ax.quiver(front_go[0], front_go[1], front_go[2],
                                     front_normal[2], front_normal[0], front_normal[1],
                                     color='blue', alpha=0.5, length=2)
        back_normal_vec = ax.quiver(back_go[0], back_go[1], back_go[2],
                                    back_normal[2], back_normal[0], back_normal[1],
                                    color='orange', alpha=0.5, length=2)

        # Markers
        for i, ln in enumerate(marker_lines):
            gp = to_graph(marker_pos_list[i][frame:frame+1])
            ln.set_data(gp[0], gp[1])
            ln.set_3d_properties(gp[2])

        ax.set_title(f'Robot Animation — t = {df.index[frame]:.3f}s')
        return (front_patch, back_patch, front_pt, back_pt, robot_com_pt,
                front_normal_vec, back_normal_vec, *marker_lines)

    base_interval = 50
    interval = int(base_interval / playback_speed)

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        interval=interval, blit=False)

    plt.tight_layout()
    plt.show()
    return ani

if __name__ == "__main__":
    df = limits.csv_to_df(limits.file_name, limits.file_dir)
    marker_1, marker_2 = recon.get_top_marker_names(df)
    dist = recon.get_marker_distance(df, marker_1, marker_2)
    flat_times = recon.get_robot_flat_times(df, dist)

    # Corrected (true) normals
    true_normals = recon.get_module_normals(df, recon.get_empirical_normal_offsets(df, flat_times))
    corrected_bottom_pts = recon.get_top_marker_bottom_points(df, true_normals)

    # Raw (uncorrected) normals from marker body plane
    front_mod_name, back_mod_name = limits.get_module_names(df)
    raw_front_normals = recon.get_marker_body_normal(df, front_mod_name)
    raw_back_normals  = recon.get_marker_body_normal(df, back_mod_name)
    raw_module_normals = (raw_front_normals, raw_back_normals)
    raw_bottom_pts = recon.get_top_marker_bottom_points(df, raw_module_normals)

    # Choose which animation to run:
    # Option 1: Robot Planes only
    # anim = animate_robot(df, corrected_bottom_pts, true_normals, playback_speed=1.0)
    # plt.show()

    # Option 2: Robot Markers only
    # anim = animate_robot_markers(df, playback_speed=1.0)
    # plt.show()

    # Option 3: Both together on the same plot, with raw planes overlaid
    animate_combined(df, corrected_bottom_pts, true_normals, playback_speed=1.0) #,
                    # raw_module_normals=raw_module_normals, raw_bottom_points=corrected_bottom_pts)

    # Option 4: PVC Markers only
    anim = animate_pvc_markers(df)
    plt.show()

    # Static plots (optional)
    # ax = recon.plot_angles(df, true_normals[0], true_normals[1])
    # recon.plot_robot_flat_times(flat_times, ax)
    # recon.plot_body_points(df, top_marker_front, top_marker_back, true_normals)
    # print(recon.get_empirical_normal_offsets(df, flat_times))
    # limits.show_plot()