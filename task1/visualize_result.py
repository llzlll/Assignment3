"""Visualize Bundle Adjustment results."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_ply(ply_path):
    """Read PLY file with colors."""
    points = []
    colors = []

    with open(ply_path, 'r') as f:
        header_lines = 0
        for line in f:
            if line.startswith('element vertex'):
                n = int(line.split()[2])
            if line == 'end_header\n':
                break
            header_lines += 1

        # Read vertices
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                colors.append([int(parts[3]), int(parts[4]), int(parts[5])])

    return np.array(points), np.array(colors) / 255.0


def read_obj(obj_path):
    """Read OBJ file with colors."""
    points = []
    colors = []

    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) >= 6:
                    points.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    colors.append([float(parts[4]), float(parts[5]), float(parts[6])])

    return np.array(points), np.array(colors)


def plot_point_cloud(file_path, output_path="point_cloud_visual.png"):
    """Plot 3D point cloud with colors."""
    # Detect file type
    if file_path.endswith('.ply'):
        points, colors = read_ply(file_path)
    else:
        points, colors = read_obj(file_path)

    print(f"Loaded {len(points)} points")
    print(f"Colors range: {colors.min():.3f} - {colors.max():.3f}")

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points with colors
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                         c=colors, s=0.5, alpha=0.6)

    # Set labels and equal aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Reconstructed 3D Point Cloud ({len(points)} points)')

    # Equal aspect ratio
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                          points[:, 1].max() - points[:, 1].min(),
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set view angles
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved point cloud visualization to {output_path}")
    plt.close()

    return points, colors


if __name__ == "__main__":
    # Visualize point cloud using PLY file (better color support)
    plot_point_cloud("reconstruction.ply")

    print("\nFiles generated:")
    print("  - reconstruction.obj   - OBJ format (some viewers may not show colors)")
    print("  - reconstruction.ply   - PLY format (better color support)")
    print("\nTo view in 3D:")
    print("  - MeshLab: https://www.meshlab.net/")
    print("  - Blender: https://www.blender.org/")
    print("  - CloudCompare: https://www.danielgm.net/cc/")
    print("  - Online: https://3dviewer.net/ (supports PLY)")
