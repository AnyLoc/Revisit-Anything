import torch
import torch.nn as nn
import os
import numpy as np


# Code from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
def log_sinkhorn_iterations(
    Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int
) -> torch.Tensor:
    """Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


# Code from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
def log_optimal_transport(
    scores: torch.Tensor, alpha: torch.Tensor, iters: int
) -> torch.Tensor:
    """Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns, bs = (m * one).to(scores), (n * one).to(scores), ((n - m) * one).to(scores)

    bins = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([scores, bins], 1)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), bs.log()[None] + norm])
    log_nu = norm.expand(n)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


class SALAD(nn.Module):
    """
    This class represents the Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.

    Attributes:
        num_channels (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int): The number of channels of the clusters (l).
        token_dim (int): The dimension of the global scene token (g).
        dropout (float): The dropout rate.
    """

    def __init__(
        self,
        num_channels=1536,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        dropout=0.3,
        expName="exp",
        reduce_feature_dims=True,  # Add flag for feature dimensionality reduction
        reduce_token_dims=True,  # Add flag for token dimensionality reduction
        args=None,
    ) -> None:
        super().__init__()

        self.expName = expName

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.reduce_feature_dims = reduce_feature_dims
        self.reduce_token_dims = reduce_token_dims
        self.args = args

        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # MLP for global scene token g
        if self.reduce_token_dims:
            self.token_features = nn.Sequential(
                nn.Linear(self.num_channels, 512),
                nn.ReLU(),
                nn.Linear(512, self.token_dim),
            )
        # MLP for local features f_i
        if self.reduce_feature_dims:
            self.cluster_features = nn.Sequential(
                nn.Conv2d(self.num_channels, 512, 1),
                dropout,
                nn.ReLU(),
                nn.Conv2d(512, self.cluster_dim, 1),
            )
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )

        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, labels=None, dirPath=None):
        """
        x (tuple): A tuple containing two elements, f and t.
            (torch.Tensor): The feature tensors (t_i) [B, C, H // 14, W // 14].
            (torch.Tensor): The token tensor (t_{n+1}) [B, C].

        Returns:
            v (torch.Tensor): The global descriptor [B, m*l + g]
        """
        x, t = x  # Extract features and token
        if self.reduce_feature_dims:
            f = self.cluster_features(x).flatten(2)
        else:
            f = x.flatten(2)
        N, D, H, W = x.shape[:]
        p = self.score(x).flatten(2)
        if self.reduce_token_dims:
            t = self.token_features(t)

        # Sinkhorn algorithm
        p = log_optimal_transport(p, self.dust_bin, 3)
        p = torch.exp(p)
        # Normalize to maintain mass
        p = p[:, :-1, :]
        if self.args.storeSOTL:
            p_otl = p.clone()

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        if self.reduce_token_dims:
            v = torch.cat(
                [
                    nn.functional.normalize(t, p=2, dim=-1),
                    nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1),
                ],
                dim=-1,
            )
            v = nn.functional.normalize(v, p=2, dim=-1)
        else:
            v = nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
        if self.args.storeSOTL:
            os.makedirs(dirPath, exist_ok=True)
            results = {}
            for i in range(N):
                image_id = labels[
                    i
                ].item()  # Assuming labels are numpy array of image IDs
                file_path = os.path.join(dirPath, f"{image_id}.npz")
                if not os.path.exists(file_path):
                    results = {
                        "soft_assign_adj": p_otl[i].cpu().numpy(),
                        "vlad": v[i].cpu().numpy(),
                    }
                    # Save the results as a .npz file
                    np.savez(file_path, **results)
        #                else:
        #                   print(f"Skipping {file_path}, already exists.")
        return v
