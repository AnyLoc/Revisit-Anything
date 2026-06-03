from pathlib import Path

import numpy as np
import torch
from loguru import logger


def build_pixel_to_patch_index(
    height: int, width: int, patch: int = 14, device=None
) -> tuple[np.ndarray, torch.Tensor]:
    """Map every pixel (i, j) to its DINO patch index.

    Returns ``(idx_matrix, ind_matrix)`` where:
      - ``idx_matrix`` is an ``int32`` array of shape ``(H, W, 2)`` holding
        the (patch_row, patch_col) for each pixel, clamped to the patch grid.
      - ``ind_matrix`` is a flat tensor of shape ``(H*W,)`` of raveled patch
        indices into a ``(dh, dw)`` grid, placed on ``device``.

    Replaces the nested ``for i, j`` loop that appeared identically in
    ``fit_pca_model`` and ``create_pairs``.
    """
    dh = height // patch
    dw = width // patch
    rows = np.minimum(np.arange(height) // patch, dh - 1)
    cols = np.minimum(np.arange(width) // patch, dw - 1)
    idx_matrix = np.empty((height, width, 2), dtype="int32")
    idx_matrix[..., 0] = rows[:, None]
    idx_matrix[..., 1] = cols[None, :]
    flat = np.ravel_multi_index((idx_matrix[..., 0].ravel(), idx_matrix[..., 1].ravel()), (dh, dw))
    ind_matrix = torch.tensor(flat, device=device)
    return idx_matrix, ind_matrix


def load_cluster_centers(vlad_cfg, device=None) -> tuple[torch.Tensor, Path]:
    """Load DINOv2 VLAD cluster centers and validate cluster count.

    ``vlad_cfg`` must expose ``cache_dir``, ``desc_layer``, ``desc_facet``,
    ``number_clusters``, and ``domain`` (same shape as ``VLADConfig`` in
    ``hydra_config.py``).
    """
    ext_specifier = f"dinov2_vitg14/l{vlad_cfg.desc_layer}_{vlad_cfg.desc_facet}_c{vlad_cfg.number_clusters}"
    c_centers_file = Path(vlad_cfg.cache_dir) / "vocabulary" / ext_specifier / vlad_cfg.domain / "c_centers.pt"
    if not c_centers_file.is_file():
        logger.error(f"Cluster centers not found: {c_centers_file}")
        raise FileNotFoundError(f"Cluster centers not found: {c_centers_file}")
    c_centers = torch.load(c_centers_file)
    if c_centers.shape[0] != vlad_cfg.number_clusters:
        raise ValueError(
            f"Number of clusters in c_centers ({c_centers.shape[0]}) "
            f"does not match vlad_config.number_clusters "
            f"({vlad_cfg.number_clusters})"
        )
    if device is not None:
        c_centers = c_centers.to(device)
    return c_centers, c_centers_file
