import reconstruction as recon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import limits

# create animation of the robot module planes moving in 3D, where the planes are defined by 
# but have the plane be defined by intersection with the corresponding top_marker_bottom_points output for that module,
# and the normal vector for the corresponding plane, derived from get_module_angles. Note that the data axes
# should correspond with the graph axes as follows: y corresponds to z, x corresponds to y, and z corresponds to x
def animate_robot(df: pd.DataFrame, top_marker_bottom_points, module_angles, playback_speed=1.0):
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    front_bottom, back_bottom = top_marker_bottom_points  # each (n_samples, 3) in data coords
    front_angles, back_angles = module_angles              # each (nx, ny, nz) in degrees

    # Reconstruct unit normals from angles (direction cosines)
    def _angles_to_normals(angle_tuple):
        x_deg, y_deg, z_deg = angle_tuple
        nx = np.cos(np.radians(x_deg))
        ny = np.cos(np.radians(y_deg))
        nz = np.cos(np.radians(z_deg))
        normals = np.column_stack((nx, ny, nz))
        # Ensure upward-pointing (positive y in data = positive z in graph)
        normals[normals[:, 1] < 0] *= -1
        return normals

    front_normals = _angles_to_normals(front_angles)  # (n_samples, 3)
    back_normals  = _angles_to_normals(back_angles)

    # Plane half-size for visualization (in data units, ~size of the module)
    plane_half_size = 0.5  # 50 cm (10x larger than original 5 cm)

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

    # Set up the figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

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
    ax.legend()

    n_frames = len(df.index)

    def init():
        front_patch.set_verts([np.zeros((4, 3))])
        back_patch.set_verts([np.zeros((4, 3))])
        front_pt.set_data([], [])
        front_pt.set_3d_properties([])
        back_pt.set_data([], [])
        back_pt.set_3d_properties([])
        return front_patch, back_patch, front_pt, back_pt

    def update(frame):
        # Front module plane
        f_corners = make_plane_patch(front_bottom[frame], front_normals[frame])
        front_patch.set_verts([f_corners])

        # Back module plane
        b_corners = make_plane_patch(back_bottom[frame], back_normals[frame])
        back_patch.set_verts([b_corners])

        # Bottom points
        fg = to_graph(front_bottom[frame:frame+1])
        bg = to_graph(back_bottom[frame:frame+1])
        front_pt.set_data(fg[0], fg[1])
        front_pt.set_3d_properties(fg[2])
        back_pt.set_data(bg[0], bg[1])
        back_pt.set_3d_properties(bg[2])

        ax.set_title(f'Robot Module Planes — t = {df.index[frame]:.3f}s')
        return front_patch, back_patch, front_pt, back_pt

    # Adjust interval based on playback speed: lower interval = faster playback
    base_interval = 50  # ms at 1x speed
    interval = int(base_interval / playback_speed)

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        interval=interval, blit=False)

    plt.tight_layout()
    plt.show()
    return ani

if __name__ == "__main__":
    df = limits.csv_to_df(limits.file_name, limits.file_dir)
    x_angle, y_angle, z_angle = recon.get_marker_body_angle(df, "FrontMod")
    marker_1, marker_2 = recon.get_top_marker_names(df)
    dist = recon.get_marker_distance(df, marker_1, marker_2)
    flat_times = recon.get_robot_flat_times(df, dist)
    true_angles = recon.get_module_angles(df, recon.get_empirical_angle_offsets(df, flat_times))

    ax = recon.plot_angles(df, true_angles[0][1], true_angles[1][1], true_angles[0][1]) # frontmod
    # ax = plot_angles(df, true_angles[1][0], true_angles[1][1], true_angles[1][2]) # backmod
    # ax = plot_marker_distance(df, dist)
    recon.plot_robot_flat_times(flat_times, ax)
    # print(f"Robot angle at first instance of flatness: {get_robot_angle_when_flat(df, flat_times)}")
    print(recon.get_empirical_angle_offsets(df, flat_times))
    # print(get_measured_angle_offsets(mod_vectors_static["front"][0], mod_vectors_static["front"][1]))
    animate_robot(df, recon.get_top_marker_bottom_points(df, true_angles), true_angles, 50)
    # limits.show_plot()