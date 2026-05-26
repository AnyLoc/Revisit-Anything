from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, TYPE_CHECKING, Union, List

import h5py
import numpy as np
import torch
from scipy.spatial import Delaunay
from tqdm import tqdm

import revisit_anything.utils as func_vpr
from revisit_anything.feature_store import FeatureStore
from revisit_anything.runtime import RuntimeCtx
import os
import einops as ein
from torch.nn import functional as F
import fast_pytorch_kmeans as fpk


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
    images: list[Path]          # length == N_images; same list passed to compute_all

    def seg_range(self, n_images: int) -> list[np.ndarray]:
        return [np.where(self.im_inds == i)[0] for i in range(n_images)]


def _build_adj_gpu(
    seg_mask_gpu: torch.Tensor, order: Optional[int]
) -> Optional[torch.Tensor]:
    """GPU port of :func:`func_vpr.nbrMasksAGGFastSingle` (utils.py:1281).

    Mask centroids are computed on GPU; the Delaunay triangulation itself
    needs scipy (no GPU equivalent in our stack), so coordinates briefly
    transit to CPU. The adjacency power is computed on GPU. Result is bool
    on the same device as ``seg_mask_gpu``.
    """
    if order is None or order < 1:
        return None

    device = seg_mask_gpu.device
    n = seg_mask_gpu.shape[0]

    # Match legacy convention: np.nonzero(mask).mean(1)[::-1] → (mean_x, mean_y).
    mask_f = seg_mask_gpu.float()
    H = mask_f.shape[-2]
    W = mask_f.shape[-1]
    rows = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
    cols = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
    total = mask_f.sum(dim=(1, 2)).clamp_min(1.0)
    mean_y = (mask_f * rows).sum(dim=(1, 2)) / total
    mean_x = (mask_f * cols).sum(dim=(1, 2)) / total
    mask_cords = torch.stack([mean_x, mean_y], dim=1).cpu().numpy()

    adj_mat = torch.zeros((n, n), device=device, dtype=torch.float32)
    if n > 3:
        tri = Delaunay(mask_cords)
        for v in range(n):
            nbrs = func_vpr.getNbrsDelaunay(tri, v)
            nbrs = np.unique([[v, v]] + nbrs)
            adj_mat[v, torch.from_numpy(nbrs).to(device)] = 1.0
        adj_power = adj_mat.clone()
        for _ in range(order - 1):
            adj_power = adj_power @ adj_mat
        return adj_power.bool()

    nbr_list = [0, 1] if n > 1 else [0]
    for v in range(n):
        for k in nbr_list:
            adj_mat[v, k] = 1.0
    return adj_mat.bool()


def seg_vlad_gpu_from_tensors(
    dino_feat_gpu: torch.Tensor,
    seg_mask_gpu: torch.Tensor,
    c_centers: torch.Tensor,
    ind_matrix: torch.Tensor,
    *,
    desired_height: int,
    desired_width: int,
    desc_dim: int = 1536,
    adj_mat: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> torch.Tensor:
    """In-memory equivalent of :func:`func_vpr.seg_vlad_gpu_single`.

    Accepts GPU-resident DINO features and segmentation masks directly so
    the fused stage can skip the H5→CPU→GPU round-trip done by the legacy
    path (``utils.py:1054``). Returns the per-segment VLAD on the requested
    device (the legacy function returns ``.cpu()``; we keep it on GPU so
    the caller can sample into a reservoir or apply PCA without another
    transfer).

    ``dino_feat_gpu`` accepts shapes ``[D, h_r, w_r]``, ``[1, D, h_r, w_r]``,
    or ``[1, D, h_r*w_r]``. It is assumed to already be on the target device.
    """
    dh = desired_height // 14
    dw = desired_width // 14

    if dino_feat_gpu.dim() == 3 and dino_feat_gpu.shape[0] == desc_dim:
        dino_desc = dino_feat_gpu.reshape(1, desc_dim, dh * dw)
    elif dino_feat_gpu.dim() == 4:
        dino_desc = dino_feat_gpu.reshape(1, desc_dim, dh * dw)
    elif dino_feat_gpu.dim() == 3 and dino_feat_gpu.shape[1] == desc_dim:
        dino_desc = dino_feat_gpu
    else:
        raise ValueError(
            f"Unexpected dino_feat shape {tuple(dino_feat_gpu.shape)}"
        )

    dino_desc_norm = F.normalize(dino_desc, dim=1)

    mask = seg_mask_gpu.float().unsqueeze(0)
    mask = F.interpolate(
        mask, [desired_height, desired_width], mode="nearest"
    ).squeeze(0).bool().reshape(seg_mask_gpu.shape[0], -1)

    n_seg = seg_mask_gpu.shape[0]
    mask_idx = torch.zeros((n_seg, dh * dw), device=device, dtype=torch.bool)
    mask_ind_flat = torch.argwhere(mask)
    mask_idx[mask_ind_flat[:, 0], ind_matrix[mask_ind_flat[:, 1]]] = True

    reg_feat_per = dino_desc_norm.permute(0, 2, 1).squeeze(0)
    c_centers_d = c_centers.to(device)
    if adj_mat is not None:
        gd, _ = func_vpr.vlad_single(
            reg_feat_per, c_centers_d, None, mask_idx, adj_mat.to(device)
        )
    else:
        gd, _ = func_vpr.vlad_single(reg_feat_per, c_centers_d, None, mask_idx)
    return gd


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
        vlads_cache_h5: Optional[Path] = None,
    ) -> AllVlads:
        """Compute all per-image VLADs in one streaming pass.

        If ``pca`` is given, PCA is applied per ``batch_size`` images so only
        the reduced-dimension result is accumulated.

        If ``vlads_cache_h5`` points to a cache built by
        :class:`FusedFeaturePcaStage`, the per-image VLAD is loaded directly
        and we only apply PCA — DINO/FastSAM/SegVLAD recomputation is skipped.
        """
        if vlads_cache_h5 is not None and Path(vlads_cache_h5).is_file():
            return self._compute_all_from_cache(
                vlads_cache_h5, images, pca, desc=desc
            )

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
        return AllVlads(feats=feats, im_inds=im_inds, images=list(images))

    def _compute_all_from_cache(
        self,
        vlads_cache_h5: Path,
        images: list[Path],
        pca: Optional["PcaModel"],
        desc: str = "SegVLAD (cached)",
    ) -> AllVlads:
        feats_list: list[torch.Tensor] = []
        im_inds_list: list[np.ndarray] = []
        with h5py.File(str(vlads_cache_h5), "r") as f:
            for i, img in enumerate(tqdm(images, desc=desc)):
                grp = f[img.name]
                vlad_np = grp["vlad"][()]
                vlad = torch.from_numpy(vlad_np).to(torch.float32)
                if pca is not None:
                    vlad = pca.transform(vlad.to(self.ctx.device))
                feats_list.append(vlad.cpu())
                im_inds_list.append(
                    np.full(vlad_np.shape[0], i, dtype=np.int64)
                )

        feats = torch.cat(feats_list, dim=0) if feats_list else torch.empty(0)
        im_inds = (
            np.concatenate(im_inds_list)
            if im_inds_list
            else np.array([], dtype=int)
        )
        return AllVlads(feats=feats, im_inds=im_inds, images=list(images))


class VLAD:
    """
        An implementation of VLAD algorithm given database and query
        descriptors.
        
        Constructor arguments:
        - num_clusters:     Number of cluster centers for VLAD
        - desc_dim:         Descriptor dimension. If None, then it is
                            inferred when running `fit` method.
        - intra_norm:       If True, intra normalization is applied
                            when constructing VLAD
        - norm_descs:       If True, the given descriptors are 
                            normalized before training and predicting 
                            VLAD descriptors. Different from the
                            `intra_norm` argument.
        - dist_mode:        Distance mode for KMeans clustering for 
                            vocabulary (not residuals). Must be in 
                            {'euclidean', 'cosine'}.
        - vlad_mode:        Mode for descriptor assignment (to cluster
                            centers) in VLAD generation. Must be in
                            {'soft', 'hard'}
        - soft_temp:        Temperature for softmax (if 'vald_mode' is
                            'soft') for assignment
        - cache_dir:        Directory to cache the VLAD vectors. If
                            None, then no caching is done. If a str,
                            then it is assumed as the folder path. Use
                            absolute paths.
        
        Notes:
        - Arandjelovic, Relja, and Andrew Zisserman. "All about VLAD."
            Proceedings of the IEEE conference on Computer Vision and 
            Pattern Recognition. 2013.
    """
    def __init__(self, num_clusters: int, 
                desc_dim: Union[int, None]=None, 
                intra_norm: bool=True, norm_descs: bool=True, 
                dist_mode: str="cosine", vlad_mode: str="hard", 
                soft_temp: float=1.0, 
                cache_dir: Union[str,None]=None) -> None:
        self.num_clusters = num_clusters
        self.desc_dim = desc_dim
        self.intra_norm = intra_norm
        self.norm_descs = norm_descs
        self.mode = dist_mode
        self.vlad_mode = str(vlad_mode).lower()
        assert self.vlad_mode in ['soft', 'hard']
        self.soft_temp = soft_temp
        # Set in the training phase
        self.c_centers = None
        self.kmeans = None
        # Set the caching
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir = os.path.abspath(os.path.expanduser(
                    self.cache_dir))
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
                print(f"Created cache directory: {self.cache_dir}")
            else:
                print("Warning: Cache directory already exists: " \
                        f"{self.cache_dir}")
        else:
            print("VLAD caching is disabled.")
    
    def can_use_cache_vlad(self):
        """
            Checks if the cache directory is a valid cache directory.
            For it to be valid, it must exist and should at least
            include the cluster centers file.
            
            Returns:
            - True if the cache directory is valid
            - False if 
                - the cache directory doesn't exist
                - exists but doesn't contain the cluster centers
                - no caching is set in constructor
        """
        if self.cache_dir is None:
            return False
        if not os.path.exists(self.cache_dir):
            return False
        if os.path.exists(f"{self.cache_dir}/c_centers.pt"):
            return True
        else:
            return False
    
    def can_use_cache_ids(self, 
                cache_ids: Union[List[str], str, None],
                only_residuals: bool=False) -> bool:
        """
            Checks if the given cache IDs exist in the cache directory
            and returns True if all of them exist.
            The cache is stored in the following files:
            - c_centers.pt:     Cluster centers
            - `cache_id`_r.pt:  Residuals for VLAD
            - `cache_id`_l.pt:  Labels for VLAD (hard assignment)
            - `cache_id`_s.pt:  Soft assignment for VLAD
            
            The function returns False if cache cannot be used or if
            any of the cache IDs are not found. If all cache IDs are
            found, then True is returned.
            
            This function is mainly for use outside the VLAD class.
        """
        if not self.can_use_cache_vlad():
            return False
        if cache_ids is None:
            return False
        if isinstance(cache_ids, str):
            cache_ids = [cache_ids]
        for cache_id in cache_ids:
            if not os.path.exists(
                    f"{self.cache_dir}/{cache_id}_r.pt"):
                return False
            if self.vlad_mode == "hard" and not os.path.exists(
                    f"{self.cache_dir}/{cache_id}_l.pt") and not \
                        only_residuals:
                return False
            if self.vlad_mode == "soft" and not os.path.exists(
                    f"{self.cache_dir}/{cache_id}_s.pt") and not \
                        only_residuals:
                return False
        return True
    
    # Generate cluster centers
    def fit(self, train_descs: Union[np.ndarray, torch.Tensor, None]):
        """
            Using the training descriptors, generate the cluster 
            centers (vocabulary). Function expects all descriptors in
            a single list (see `fit_and_generate` for a batch of 
            images).
            If the cache directory is valid, then retrieves cluster
            centers from there (the `train_descs` are ignored). 
            Otherwise, stores the cluster centers in the cache 
            directory (if using caching).
            
            Parameters:
            - train_descs:  Training descriptors of shape 
                            [num_train_desc, desc_dim]. If None, then
                            caching should be valid (else ValueError).
        """
        # Clustering to create vocabulary
        self.kmeans = fpk.KMeans(self.num_clusters, mode=self.mode)
        # Check if cache exists
        if self.can_use_cache_vlad():
            print("Using cached cluster centers")
            self.c_centers = torch.load(
                    f"{self.cache_dir}/c_centers.pt")
            self.kmeans.centroids = self.c_centers
            if self.desc_dim is None:
                self.desc_dim = self.c_centers.shape[1]
                print(f"Desc dim set to {self.desc_dim}")
        else:
            if train_descs is None:
                raise ValueError("No training descriptors given")
            if type(train_descs) == np.ndarray:
                train_descs = torch.from_numpy(train_descs).\
                    to(torch.float32)
            if self.desc_dim is None:
                self.desc_dim = train_descs.shape[1]
            if self.norm_descs:
                train_descs = F.normalize(train_descs)
            self.kmeans.fit(train_descs)
            self.c_centers = self.kmeans.centroids
            if self.cache_dir is not None:
                print("Caching cluster centers")
                torch.save(self.c_centers, 
                        f"{self.cache_dir}/c_centers.pt")
    
    def fit_and_generate(self, 
                train_descs: Union[np.ndarray, torch.Tensor]) \
                -> torch.Tensor:
        """
            Given a batch of descriptors over images, `fit` the VLAD
            and generate the global descriptors for the training
            images. Use only when there are a fixed number of 
            descriptors in each image.
            
            Parameters:
            - train_descs:  Training image descriptors of shape
                            [num_imgs, num_descs, desc_dim]. There are
                            'num_imgs' images, each image has 
                            'num_descs' descriptors and each 
                            descriptor is 'desc_dim' dimensional.
            
            Returns:
            - train_vlads:  The VLAD vectors of all training images.
                            Shape: [num_imgs, num_clusters*desc_dim]
        """
        # Generate vocabulary
        all_descs = ein.rearrange(train_descs, "n k d -> (n k) d")
        self.fit(all_descs)
        # For each image, stack VLAD
        return torch.stack([self.generate(tr) for tr in train_descs])
    
    def generate(self, query_descs: Union[np.ndarray, torch.Tensor],
                cache_id: Union[str, None]=None) -> torch.Tensor:
        """
            Given the query descriptors, generate a VLAD vector. Call
            `fit` before using this method. Use this for only single
            images and with descriptors stacked. Use function
            `generate_multi` for multiple images.
            
            Parameters:
            - query_descs:  Query descriptors of shape [n_q, desc_dim]
                            where 'n_q' is number of 'desc_dim' 
                            dimensional descriptors in a query image.
            - cache_id:     If not None, then the VLAD vector is
                            constructed using the residual and labels
                            from this file.
            
            Returns:
            - n_vlas:   Normalized VLAD: [num_clusters*desc_dim]
        """
        residuals = self.generate_res_vec(query_descs, cache_id)
        # Un-normalized VLAD vector: [c*d,]
        un_vlad = torch.zeros(self.num_clusters * self.desc_dim)
        if self.vlad_mode == 'hard':
            # Get labels for assignment of descriptors
            if cache_id is not None and self.can_use_cache_vlad() \
                    and os.path.isfile(
                        f"{self.cache_dir}/{cache_id}_l.pt"):
                labels = torch.load(
                        f"{self.cache_dir}/{cache_id}_l.pt")
            else:
                labels = self.kmeans.predict(query_descs)   # [q]
                if cache_id is not None and self.can_use_cache_vlad():
                    torch.save(labels, 
                            f"{self.cache_dir}/{cache_id}_l.pt")
            # Create VLAD from residuals and labels
            used_clusters = set(labels.numpy())
            for k in used_clusters:
                # Sum of residuals for the descriptors in the cluster
                #  Shape:[q, c, d]  ->  [q', d] -> [d]
                cd_sum = residuals[labels==k,k].sum(dim=0)
                if self.intra_norm:
                    cd_sum = F.normalize(cd_sum, dim=0)
                un_vlad[k*self.desc_dim:(k+1)*self.desc_dim] = cd_sum
        else:       # Soft cluster assignment
            # Cosine similarity: 1 = close, -1 = away
            if cache_id is not None and self.can_use_cache_vlad() \
                    and os.path.isfile(
                        f"{self.cache_dir}/{cache_id}_s.pt"):
                soft_assign = torch.load(
                        f"{self.cache_dir}/{cache_id}_s.pt")
            else:
                cos_sims = F.cosine_similarity( # [q, c]
                        ein.rearrange(query_descs, "q d -> q 1 d"), 
                        ein.rearrange(self.c_centers, "c d -> 1 c d"), 
                        dim=2)
                soft_assign = F.softmax(self.soft_temp*cos_sims, 
                        dim=1)
                if cache_id is not None and self.can_use_cache_vlad():
                    torch.save(soft_assign, 
                            f"{self.cache_dir}/{cache_id}_s.pt")
            # Soft assignment scores (as probabilities): [q, c]
            for k in range(0, self.num_clusters):
                w = ein.rearrange(soft_assign[:, k], "q -> q 1 1")
                # Sum of residuals for all descriptors (for cluster k)
                cd_sum = ein.rearrange(w * residuals, 
                            "q c d -> (q c) d").sum(dim=0)  # [d]
                if self.intra_norm:
                    cd_sum = F.normalize(cd_sum, dim=0)
                un_vlad[k*self.desc_dim:(k+1)*self.desc_dim] = cd_sum
        # Normalize the VLAD vector
        n_vlad = F.normalize(un_vlad, dim=0)
        return n_vlad
    
    def generate_multi(self, 
            multi_query: Union[np.ndarray, torch.Tensor, list],
            cache_ids: Union[List[str], None]=None) \
            -> Union[torch.Tensor, list]:
        """
            Given query descriptors from multiple images, generate
            the VLAD for them.
            
            Parameters:
            - multi_query:  Descriptors of shape [n_imgs, n_kpts, d]
                            There are 'n_imgs' and each image has
                            'n_kpts' keypoints, with 'd' dimensional
                            descriptor each. If a List (can then have
                            different number of keypoints in each 
                            image), then the result is also a list.
            - cache_ids:    Cache IDs for the VLAD vectors. If None,
                            then no caching is done (stored or 
                            retrieved). If a list, then the length
                            should be 'n_imgs' (one per image).
            
            Returns:
            - multi_res:    VLAD descriptors for the queries
        """
        if cache_ids is None:
            cache_ids = [None] * len(multi_query)
        res = [self.generate(q, c) \
                for (q, c) in zip(multi_query, cache_ids)]
        try:    # Most likely pytorch
            res = torch.stack(res)
        except TypeError:
            try:    # Otherwise numpy
                res = np.stack(res)
            except TypeError:
                pass    # Let it remain as a list
        return res
    
    def generate_res_vec(self, 
                query_descs: Union[np.ndarray, torch.Tensor],
                cache_id: Union[str, None]=None) -> torch.Tensor:
        """
            Given the query descriptors, generate a VLAD vector. Call
            `fit` before using this method. Use this for only single
            images and with descriptors stacked. Use function
            `generate_multi` for multiple images.
            
            Parameters:
            - query_descs:  Query descriptors of shape [n_q, desc_dim]
                            where 'n_q' is number of 'desc_dim' 
                            dimensional descriptors in a query image.
            - cache_id:     If not None, then the VLAD vector is
                            constructed using the residual and labels
                            from this file.
            
            Returns:
            - residuals:    Residual vector: shape [n_q, n_c, d]
        """
        assert self.kmeans is not None
        assert self.c_centers is not None
        # Compute residuals (all query to cluster): [q, c, d]
        if cache_id is not None and self.can_use_cache_vlad() and \
                os.path.isfile(f"{self.cache_dir}/{cache_id}_r.pt"):
            residuals = torch.load(
                    f"{self.cache_dir}/{cache_id}_r.pt")
        else:
            if type(query_descs) == np.ndarray:
                query_descs = torch.from_numpy(query_descs)\
                    .to(torch.float32)
            if self.norm_descs:
                query_descs = F.normalize(query_descs)
            residuals = ein.rearrange(query_descs, "q d -> q 1 d") \
                    - ein.rearrange(self.c_centers, "c d -> 1 c d")
            if cache_id is not None and self.can_use_cache_vlad():
                cid_dir = f"{self.cache_dir}/"\
                        f"{os.path.split(cache_id)[0]}"
                if not os.path.isdir(cid_dir):
                    os.makedirs(cid_dir)
                    print(f"Created directory: {cid_dir}")
                torch.save(residuals, 
                        f"{self.cache_dir}/{cache_id}_r.pt")
        # print("residuals",residuals.shape)
        return residuals

    def generate_multi_res_vec(self, 
            multi_query: Union[np.ndarray, torch.Tensor, list],
            cache_ids: Union[List[str], None]=None) \
            -> Union[torch.Tensor, list]:
        """
            Given query descriptors from multiple images, generate
            the VLAD for them.
            
            Parameters:
            - multi_query:  Descriptors of shape [n_imgs, n_kpts, d]
                            There are 'n_imgs' and each image has
                            'n_kpts' keypoints, with 'd' dimensional
                            descriptor each. If a List (can then have
                            different number of keypoints in each 
                            image), then the result is also a list.
            - cache_ids:    Cache IDs for the VLAD vectors. If None,
                            then no caching is done (stored or 
                            retrieved). If a list, then the length
                            should be 'n_imgs' (one per image).
                            
            Returns:
            - multi_res:    VLAD descriptors for the queries
        """
        if cache_ids is None:
            cache_ids = [None] * len(multi_query)
        res = [self.generate_res_vec(q, c) \
                for (q, c) in zip(multi_query, cache_ids)]
        try:    # Most likely pytorch
            res = torch.stack(res)
        except TypeError:
            try:    # Otherwise numpy
                res = np.stack(res)
            except TypeError:
                pass    # Let it remain as a list
        return res
