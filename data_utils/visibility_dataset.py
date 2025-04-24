import numpy as np
import torch
from torch.utils.data import Dataset

class VisibilityDataset(Dataset):
    """
    Dataset for sampling fixed-size angular patches from a point cloud.

    Input (.xyz text or .npy) with columns: x, y, z, u, v, label.
      - (u,v) pixel coords on 1920×1080 spherical image
      - label: 0 (occluded) or 1 (visible)

    Returns per sample tuple:
      feats:  Tensor (num_point, 5)  => [dtheta, dphi, r_norm, u_norm, v_norm]
      theta:  Tensor (num_point,)    => absolute θ for each point
      phi:    Tensor (num_point,)    => absolute φ for each point
      labels: LongTensor (num_point,)
    """
    def __init__(self, file_path, num_point=4096, ang_radius=0.05, sample_rate=1.0):
        super().__init__()
        # Load raw
        if file_path.endswith('.npy'):
            data = np.load(file_path)
        else:
            data = np.loadtxt(file_path)
        assert data.shape[1] == 6, f"Expected 6 cols, got {data.shape}"

        self.xyz    = data[:, :3].astype(np.float32)
        self.uv     = data[:, 3:5].astype(np.float32)
        self.labels = data[:, 5].astype(np.int64)
        self.N      = len(self.xyz)

        # Spherical coords
        x, y, z = self.xyz.T
        self.r     = np.linalg.norm(self.xyz, axis=1).astype(np.float32)
        self.theta = np.arctan2(y, x).astype(np.float32)
        self.phi   = np.arccos(np.clip(z / (self.r + 1e-8), -1, 1)).astype(np.float32)

        # Normalize uv to [0,1]
        self.u_norm = (self.uv[:, 0] / 1919.0).astype(np.float32)
        self.v_norm = (self.uv[:, 1] / 1079.0).astype(np.float32)

        # Sampling
        self.num_point  = num_point
        self.ang_radius = ang_radius
        total_samples   = max(1, int(self.N * sample_rate / num_point))
        self._len       = total_samples

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # pick random centroid
        while True:
            i = np.random.randint(0, self.N)
            ct, cp = self.theta[i], self.phi[i]
            dtheta = self.theta - ct
            dphi   = self.phi   - cp
            ang_d  = np.sqrt(dtheta**2 + dphi**2)
            nbrs   = np.where(ang_d <= self.ang_radius)[0]
            if len(nbrs) > 0:
                break

        if len(nbrs) >= self.num_point:
            idxs = np.random.choice(nbrs, self.num_point, replace=False)
        else:
            idxs = np.random.choice(nbrs, self.num_point, replace=True)

        rel_dtheta = dtheta[idxs]
        rel_dphi   = dphi[idxs]
        r_norm     = self.r[idxs] / (self.r.max() + 1e-8)
        u_n        = self.u_norm[idxs]
        v_n        = self.v_norm[idxs]

        feats = np.stack([rel_dtheta, rel_dphi, r_norm, u_n, v_n], axis=1)
        theta_c = self.theta[idxs]
        phi_c   = self.phi[idxs]
        labs    = self.labels[idxs]

        return (
            torch.from_numpy(feats),    # (num_point,5)
            torch.from_numpy(theta_c),   # (num_point,)
            torch.from_numpy(phi_c),     # (num_point,)
            torch.from_numpy(labs)       # (num_point,)
        )
