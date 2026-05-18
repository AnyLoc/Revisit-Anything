from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

import func_vpr
from revisit_anything.feature_store import FeatureStore
from revisit_anything.runtime import RuntimeCtx

if TYPE_CHECKING:
    from revisit_anything.pca_model import PcaModel


@dataclass
class ImageVlad:
    img_idx: int
    vlad: torch.Tensor          # [n_seg, vlad_dim] on CPU after seg_vlad_gpu_single.cpu()
    im_inds: np.ndarray         # [n_seg] all equal to img_idx


@dataclass
class AllVlads:
    feats: torch.Tensor         # [N_seg_total, dim] (post-PCA if pca was given)
    im_inds: np.ndarray         # [N_seg_total]

    def seg_range(self, n_images: int) -> list[np.ndarray]:
        return [np.where(self.im_inds == i)[0] for i in range(n_images)]


class VladComputer:
    """Per-image SegVLAD computation, reused by PCA fit (Stage 2) and pair
    construction (Stage 3). Holds device-resident lookup tensors so they are
    not rebuilt or reuploaded across calls.
    """

    def __init__(
        self,
        ctx: RuntimeCtx,
        c_centers: torch.Tensor,
        ind_matrix: torch.Tensor,
        idx_matrix: np.ndarray,
        *,
        desired_width: int,
        desired_height: int,
        desc_dim: int = 1536,
        order: Optional[int] = 3,
        min_area: int = 0,
    ):
        self.ctx = ctx
        self.c_centers = c_centers.to(ctx.device)
        self.ind_matrix = ind_matrix.to(ctx.device)
        self.idx_matrix = idx_matrix
        self.desc_dim = desc_dim
        self.order = order
        self.min_area = min_area
        # func_vpr.seg_vlad_gpu_single only consumes desired_width/height from cfg.
        self._legacy_cfg = {
            "desired_width": desired_width,
            "desired_height": desired_height,
        }

    def compute_one(
        self, store: FeatureStore, img_name: str, img_idx: int
    ) -> ImageVlad:
        masks_seg = func_vpr.preload_masks(store.masks_h5, img_name)
        im_inds, _, seg_mask = func_vpr.getIdxSingleFast(
            img_idx, masks_seg, minArea=self.min_area
        )
        adj_mat = (
            func_vpr.nbrMasksAGGFastSingle(masks_seg, self.order)
            if self.order
            else None
        )
        gd = func_vpr.seg_vlad_gpu_single(
            self.ind_matrix, self.idx_matrix, store.dino_h5, img_name,
            seg_mask, self.c_centers, self._legacy_cfg,
            desc_dim=self.desc_dim, adj_mat=adj_mat,
            device=str(self.ctx.device),
        )
        return ImageVlad(img_idx=img_idx, vlad=gd.to(torch.float32), im_inds=im_inds)

    def iter(
        self, store: FeatureStore, images: list[Path], desc: str = "SegVLAD"
    ) -> Iterator[ImageVlad]:
        for i, img in enumerate(tqdm(images, desc=desc)):
            yield self.compute_one(store, img.name, i)

    def compute_all(
        self,
        store: FeatureStore,
        images: list[Path],
        pca: Optional["PcaModel"] = None,
        batch_size: int = 100,
        desc: str = "SegVLAD",
    ) -> AllVlads:
        """Compute all per-image VLADs in one streaming pass.

        If ``pca`` is given, PCA is applied per ``batch_size`` images so only
        the reduced-dimension result is accumulated.
        """
        feats_list: list[torch.Tensor] = []
        batch_buf: list[torch.Tensor] = []
        im_inds_list: list[np.ndarray] = []

        for k, item in enumerate(self.iter(store, images, desc=desc), start=1):
            im_inds_list.append(item.im_inds)
            if pca is None:
                feats_list.append(item.vlad)
                continue
            batch_buf.append(item.vlad)
            if k % batch_size == 0 or k == len(images):
                batch = torch.cat(batch_buf, dim=0)
                batch_buf = []
                feats_list.append(pca.transform(batch))

        feats = torch.cat(feats_list, dim=0) if feats_list else torch.empty(0)
        im_inds = (
            np.concatenate(im_inds_list)
            if im_inds_list
            else np.array([], dtype=int)
        )
        return AllVlads(feats=feats, im_inds=im_inds)
