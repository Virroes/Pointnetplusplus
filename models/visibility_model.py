import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Angular PointNet++ Modules ---

def angular_ball_query(theta, phi, cent_theta, cent_phi, radius):
    """
    Return neighbor indices for each centroid based on angular radius.
    theta, phi: (N,)
    cent_theta, cent_phi: (M,)
    returns list of length M, each an array of neighbor indices
    """
    # Compute pairwise angular distances
    # Expand dims to (M,N)
    dtheta = theta[None, :] - cent_theta[:, None]
    dphi   = phi[None, :]   - cent_phi  [:, None]
    dist = torch.sqrt(dtheta**2 + dphi**2)
    # neighbor mask
    mask = dist <= radius
    neighbor_idx = [torch.nonzero(mask[i]).squeeze(1) for i in range(mask.shape[0])]
    return neighbor_idx

class AngularSetAbstraction(nn.Module):
    def __init__(self, n_centroids, radius, nsample, in_channels, mlp_channels):
        super().__init__()
        self.n_centroids = n_centroids
        self.radius = radius
        self.nsample = nsample
        self.mlp = nn.Sequential()
        last_ch = in_channels
        for i, out_ch in enumerate(mlp_channels):
            self.mlp.add_module(f"conv{i}", nn.Conv2d(last_ch, out_ch, 1))
            self.mlp.add_module(f"bn{i}", nn.BatchNorm2d(out_ch))
            self.mlp.add_module(f"relu{i}", nn.ReLU(inplace=True))
            last_ch = out_ch

    def forward(self, theta, phi, features):
        # theta, phi: (B,N)
        B, N = theta.shape
        device = theta.device
        # 1) FPS in angular domain
        # Use random sampling for simplicity
        cent_idx = torch.randperm(N, device=device)[:self.n_centroids]
        cent_theta = theta[:, cent_idx]  # (B, M)
        cent_phi   = phi  [:, cent_idx]
        # 2) Ball query
        grouped_feats = []
        for b in range(B):
            neigh_idxs = angular_ball_query(theta[b], phi[b], cent_theta[b], cent_phi[b], self.radius)
            # For each centroid, gather up to nsample neighbors
            feats_b = []
            for idxs in neigh_idxs:
                if idxs.numel() >= self.nsample:
                    chosen = idxs[torch.randperm(idxs.numel(), device=device)[:self.nsample]]
                else:
                    chosen = idxs[torch.randint(0, idxs.numel(), (self.nsample,), device=device)]
                feats = features[b:b+1, :, chosen]  # (1, C, nsample)
                feats_b.append(feats)
            feats_b = torch.stack(feats_b, dim=-1)  # (1, C, nsample, M)
            grouped_feats.append(feats_b)
        grouped_feats = torch.cat(grouped_feats, dim=0)  # (B, C, nsample, M)
        # 3) MLP + max-pool
        new_feats = self.mlp(grouped_feats)            # (B, C', nsample, M)
        new_feats = torch.max(new_feats, dim=2)[0]      # (B, C', M)
        return cent_theta, cent_phi, new_feats

class AngularFeaturePropagation(nn.Module):
    def __init__(self, in_channels, mlp_channels):
        super().__init__()
        self.mlp = nn.Sequential()
        last_ch = in_channels
        for i, out_ch in enumerate(mlp_channels):
            self.mlp.add_module(f"conv{i}", nn.Conv1d(last_ch, out_ch, 1))
            self.mlp.add_module(f"bn{i}", nn.BatchNorm1d(out_ch))
            self.mlp.add_module(f"relu{i}", nn.ReLU(inplace=True))
            last_ch = out_ch

    def forward(self, low_theta, low_phi, low_feats, high_theta, high_phi, high_feats):
        # low: (B, C1, N), high: (B, C2, M)
        # Interpolate high_feats onto low positions via inverse angular distance
        # Simplify: use nearest neighbor
        B, _, N = low_feats.shape
        _, C2, M = high_feats.shape
        device = low_feats.device
        dtheta = low_theta.unsqueeze(-1) - high_theta.unsqueeze(-2)  # (B, N, M)
        dphi   = low_phi  .unsqueeze(-1) - high_phi  .unsqueeze(-2)
        dist = torch.sqrt(dtheta**2 + dphi**2)  # (B, N, M)
        idx = torch.argmin(dist, dim=-1)        # (B, N)
        interp = high_feats.permute(0,2,1)      # (B, M, C2)
        interpolated = torch.gather(interp, 1, idx.unsqueeze(-1).expand(-1,-1,C2))  # (B, N, C2)
        interpolated = interpolated.permute(0,2,1)  # (B, C2, N)

        # Concatenate low_feats
        if low_feats is not None:
            cat = torch.cat([low_feats, interpolated], dim=1)
        else:
            cat = interpolated
        return self.mlp(cat)

# --- Model Definition ---

class VisibilityPointNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Inputs: features of dim=5 (dtheta,dphi,r_norm,u,v)
        self.sa1 = AngularSetAbstraction(1024, 0.05, 32, in_channels=5, mlp_channels=[32,32,64])
        self.sa2 = AngularSetAbstraction(256,  0.1, 32, in_channels=64, mlp_channels=[64,64,128])
        self.sa3 = AngularSetAbstraction(64,   0.2, 32, in_channels=128, mlp_channels=[128,128,256])
        self.sa4 = AngularSetAbstraction(16,   0.4, 32, in_channels=256, mlp_channels=[256,256,512])

        self.fp4 = AngularFeaturePropagation(in_channels=512+256, mlp_channels=[256,256])
        self.fp3 = AngularFeaturePropagation(in_channels=256+128, mlp_channels=[256,256])
        self.fp2 = AngularFeaturePropagation(in_channels=256+64,  mlp_channels=[256,128])
        self.fp1 = AngularFeaturePropagation(in_channels=128+5,   mlp_channels=[128,128,128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1   = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, feats, theta, phi):
        # feats: (B,5,N), theta/phi: (B,N)
        bt, bp, f1 = self.sa1(theta, phi, feats)
        bt2, bp2, f2 = self.sa2(bt, bp, f1)
        bt3, bp3, f3 = self.sa3(bt2, bp2, f2)
        bt4, bp4, f4 = self.sa4(bt3, bp3, f3)

        up3 = self.fp4(bt3, bp3, f3, bt4, bp4, f4)
        up2 = self.fp3(bt2, bp2, f2, bt3, bp3, up3)
        up1 = self.fp2(bt , bp , f1, bt2, bp2, up2)
        up0 = self.fp1(theta, phi, feats, bt, bp, up1)

        x = F.relu(self.bn1(self.conv1(up0)))
        x = self.drop1(x)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        return x  # (B,2,N)

if __name__ == '__main__':
    import torch
    B,N = 2,4096
    feats = torch.randn(B,5,N)
    theta = torch.randn(B,N)
    phi   = torch.randn(B,N)
    model = VisibilityPointNet(num_classes=2)
    out = model(feats, theta, phi)
    print(out.shape)  # should be (B,2,N)
