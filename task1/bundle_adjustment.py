"""
Bundle Adjustment implementation with PyTorch
Optimizes: focal length, camera extrinsics (R, T), and 3D point coordinates
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def euler_angles_to_matrix(euler_angles, convention="XYZ"):
    """
    Convert Euler angles to rotation matrix.

    Args:
        euler_angles: (..., 3) Euler angles [alpha, beta, gamma] in radians
        convention: rotation convention, e.g., "XYZ", "ZYX", etc.

    Returns:
        (..., 3, 3) rotation matrices
    """
    # Extract angles
    alpha = euler_angles[..., 0]
    beta = euler_angles[..., 1]
    gamma = euler_angles[..., 2]

    # Compute sin and cos
    ca, cb, cg = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sa, sb, sg = torch.sin(alpha), torch.sin(beta), torch.sin(gamma)

    # Build rotation matrix based on convention
    zeros = torch.zeros_like(alpha)
    ones = torch.ones_like(alpha)

    if convention == "XYZ":
        # Rx @ Ry @ Rz
        Rx = torch.stack([
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, ca, -sa], dim=-1),
            torch.stack([zeros, sa, ca], dim=-1),
        ], dim=-2)

        Ry = torch.stack([
            torch.stack([cb, zeros, sb], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sb, zeros, cb], dim=-1),
        ], dim=-2)

        Rz = torch.stack([
            torch.stack([cg, -sg, zeros], dim=-1),
            torch.stack([sg, cg, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ], dim=-2)

        return Rx @ Ry @ Rz
    else:
        raise ValueError(f"Convention {convention} not implemented")


# Constants
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
CX = IMAGE_WIDTH / 2
CY = IMAGE_HEIGHT / 2
NUM_VIEWS = 50
NUM_POINTS = 20000


class Points2DDataset(Dataset):
    """Dataset for 2D point observations across multiple views."""

    def __init__(self, npz_path):
        self.data = np.load(npz_path)
        self.view_keys = sorted([k for k in self.data.keys() if k.startswith('view_')])

    def __len__(self):
        return len(self.view_keys)

    def __getitem__(self, idx):
        key = self.view_keys[idx]
        obs = torch.tensor(self.data[key], dtype=torch.float32)  # (20000, 3): [x, y, visibility]
        return obs


def project_points(points3d, R, T, f):
    """
    Project 3D points to 2D image coordinates.

    Args:
        points3d: (N, 3) 3D point coordinates [X, Y, Z]
        R: (3, 3) rotation matrix
        T: (3,) translation vector
        f: scalar focal length

    Returns:
        (N, 2) 2D projected coordinates [u, v]
    """
    # Transform to camera coordinates: [Xc, Yc, Zc] = R @ P + T
    points_cam = points3d @ R.T + T  # (N, 3)

    # Perspective projection
    # u = -f * Xc/Zc + cx  (negative for correct orientation)
    # v = f * Yc/Zc + cy
    epsilon = 1e-8
    u = -f * points_cam[:, 0] / (points_cam[:, 2] + epsilon) + CX
    v = f * points_cam[:, 1] / (points_cam[:, 2] + epsilon) + CY

    return torch.stack([u, v], dim=1)  # (N, 2)


def compute_loss(points2d_pred, points2d_obs, visibility):
    """
    Compute reprojection loss, weighted by visibility.

    Args:
        points2d_pred: (N, 2) predicted 2D coordinates
        points2d_obs: (N, 3) observed 2D coordinates [x, y, visibility]
        visibility: (N,) visibility mask

    Returns:
        scalar loss
    """
    # L2 loss for visible points only
    mask = (visibility > 0.5).float()
    diff = (points2d_pred - points2d_obs[:, :2]) ** 2
    loss = torch.sum(diff * mask.unsqueeze(1)) / (torch.sum(mask) + epsilon())
    return loss


def epsilon():
    return 1e-8


class BundleAdjustment(nn.Module):
    """Bundle Adjustment model optimizing camera parameters and 3D points."""

    def __init__(self, num_views, num_points, init_distance=2.5):
        super().__init__()
        self.num_views = num_views
        self.num_points = num_points

        # Focal length (shared across all cameras)
        # Initialize based on FoV ~ 60 degrees
        f_init = IMAGE_HEIGHT / (2 * np.tan(np.radians(60) / 2))
        self.f = nn.Parameter(torch.tensor([f_init], dtype=torch.float32))

        # Camera extrinsics
        # Initialize R to identity (Euler angles = 0)
        self.euler_angles = nn.Parameter(torch.zeros(num_views, 3, dtype=torch.float32))

        # Initialize T to [0, 0, -d] (camera in front of object at distance d)
        T_init = torch.zeros(num_views, 3, dtype=torch.float32)
        T_init[:, 2] = -init_distance
        self.T = nn.Parameter(T_init)

        # 3D point coordinates
        # Initialize to random positions near origin
        self.points3d = nn.Parameter(torch.randn(num_points, 3, dtype=torch.float32) * 0.5)

    def get_rotation_matrices(self):
        """Convert Euler angles to rotation matrices."""
        return euler_angles_to_matrix(self.euler_angles, convention="XYZ")  # (N_views, 3, 3)

    def forward(self, view_idx):
        """
        Compute projected 2D points for a specific view.

        Args:
            view_idx: index of the view (0 to num_views-1)

        Returns:
            (N_points, 2) 2D projected coordinates
        """
        R = self.get_rotation_matrices()[view_idx]  # (3, 3)
        T = self.T[view_idx]  # (3,)
        f = self.f[0]

        return project_points(self.points3d, R, T, f)

    def predict_all_views(self):
        """Predict 2D points for all views."""
        R_all = self.get_rotation_matrices()  # (N_views, 3, 3)
        T_all = self.T  # (N_views, 3)
        f = self.f[0]

        predictions = []
        for i in range(self.num_views):
            pred = project_points(self.points3d, R_all[i], T_all[i], f)
            predictions.append(pred)

        return torch.stack(predictions)  # (N_views, N_points, 2)


def train(model, dataset, num_epochs=1000, lr=1e-2, device='cpu', verbose=True):
    """
    Train the Bundle Adjustment model.

    Args:
        model: BundleAdjustment model
        dataset: Points2DDataset
        num_epochs: number of training epochs
        lr: learning rate
        device: 'cpu' or 'cuda'
        verbose: print progress

    Returns:
        list of loss values per epoch
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)

    losses = []

    # Preload all observations to device
    obs_all = [dataset[i].to(device) for i in range(len(dataset))]
    vis_all = [obs[:, 2] for obs in obs_all]

    for epoch in range(num_epochs):
        # Forward pass for all views at once
        pred_all = model.predict_all_views()  # (N_views, N_points, 2)

        # Compute loss for each view and sum
        losses_per_view = []
        for view_idx in range(len(dataset)):
            obs = obs_all[view_idx]
            visibility = vis_all[view_idx]
            pred = pred_all[view_idx]
            loss = compute_loss(pred, obs, visibility)
            losses_per_view.append(loss)

        total_loss = sum(losses_per_view) / len(dataset)
        losses.append(total_loss.item())

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        scheduler.step(total_loss.item())

        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            lr_current = optimizer.param_groups[0]['lr']
            f_current = model.f[0].item()
            print(f"Epoch {epoch:4d}: loss={total_loss.item():.6f}, f={f_current:.2f}, lr={lr_current:.6f}")

    return losses


def plot_losses(losses, output_path="loss_curve.png"):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Bundle Adjustment Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved loss curve to {output_path}")
    plt.close()


def save_obj(points3d, colors, output_path="reconstruction.obj"):
    """
    Save 3D point cloud as OBJ file with colors.

    Args:
        points3d: (N, 3) or (N, 3) numpy array of 3D coordinates
        colors: (N, 3) numpy array of RGB colors (assumed to be in [0, 1] range)
        output_path: output OBJ file path
    """
    points3d = points3d.detach().cpu().numpy()

    # Colors assumed to be already in [0, 1] range
    with open(output_path, 'w') as f:
        f.write("# Point cloud with colors (r g b)\n")
        for i in range(len(points3d)):
            x, y, z = points3d[i]
            r, g, b = colors[i]
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")

    print(f"Saved point cloud to {output_path}")


def save_ply(points3d, colors, output_path="reconstruction.ply"):
    """
    Save 3D point cloud as PLY file with colors (more widely supported).

    Args:
        points3d: (N, 3) numpy array of 3D coordinates
        colors: (N, 3) numpy array of RGB colors in [0, 1] range
        output_path: output PLY file path
    """
    points3d = points3d.detach().cpu().numpy()

    # Convert colors to [0, 255] for PLY
    colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    n = len(points3d)

    with open(output_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Vertex data
        for i in range(n):
            x, y, z = points3d[i]
            r, g, b = colors_uint8[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    print(f"Saved point cloud to {output_path}")


def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    dataset = Points2DDataset("data/points2d.npz")
    points3d_colors = np.load("data/points3d_colors.npy")
    print(f"Loaded {len(dataset)} views, {NUM_POINTS} points")

    # Initialize model
    print("\nInitializing model...")
    model = BundleAdjustment(num_views=NUM_VIEWS, num_points=NUM_POINTS, init_distance=2.5)

    # Train
    print("\nTraining...")
    losses = train(model, dataset, num_epochs=1500, lr=1e-2, device=device)

    # Plot loss curve
    print("\nPlotting loss curve...")
    plot_losses(losses, "loss_curve.png")

    # Save reconstruction (both OBJ and PLY formats)
    print("\nSaving reconstruction...")
    save_obj(model.points3d, points3d_colors, "reconstruction.obj")
    save_ply(model.points3d, points3d_colors, "reconstruction.ply")

    # Print final parameters
    print(f"\nFinal focal length: {model.f[0].item():.4f}")
    print(f"First camera T: {model.T[0].detach().cpu().numpy()}")
    print(f"First camera Euler angles: {model.euler_angles[0].detach().cpu().numpy()}")

    # Compute final statistics
    with torch.no_grad():
        model = model.to(device)
        total_error = 0.0
        total_visible = 0
        for view_idx in range(len(dataset)):
            obs = dataset[view_idx].to(device)
            visibility = obs[:, 2]
            pred = model(view_idx)
            mask = (visibility > 0.5).float()
            error = torch.sqrt(torch.sum((pred - obs[:, :2])**2, dim=1))
            total_error += (error * mask).sum().item()
            total_visible += mask.sum().item()

        avg_error = total_error / total_visible
        print(f"\nAverage reprojection error: {avg_error:.4f} pixels")


if __name__ == "__main__":
    main()
