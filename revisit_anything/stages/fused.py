from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from revisit_anything.config import Config
import torch
import torchvision.transforms as tvf
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import revisit_anything.utils as func_vpr
from revisit_anything.feature_store import FeatureStore
from revisit_anything.h5_writer import (
    AsyncH5Writer,
    dino_write_fn,
    masks_write_fn,
    vlads_write_fn,
)
from revisit_anything.pca_model import PcaModel
from revisit_anything.runtime import RuntimeCtx
from revisit_anything.stages.pca import PcaFitter
from revisit_anything.stages.vlad import (
    _build_adj_gpu,
    seg_vlad_gpu_from_tensors,
)
from revisit_anything.timer import Timer


# Mirrors ``extract._legacy_cfg`` so the FastSAM helpers see the dict shape
# they were originally written against. Duplicated rather than imported to
# keep stages independent.
def _legacy_cfg(width: int, height: int) -> dict:
    return {
        "desired_width": width,
        "desired_height": height,
        "detect": "dino",
        "use_sam": True,
        "class_threshold": 0.9,
        "desired_feature": 0,
        "query_type": "text",
        "sort_by": "area",
        "use_16bit": False,
        "use_cuda": True,
        "dino_strides": 4,
        "use_traced_model": False,
        "rmin": 0,
        "DAStoreFull": False,
        "dinov2": True,
        "wrap": False,
        "resize": True,
    }


class _BatchDataset(Dataset):
    """Reads + preprocesses one image. Returns both the DINO-ready tensor
    and the BGR resized array used by ``process_batch_FastSAM``.

    Centralising the read/resize so each image is decoded once instead of
    twice (legacy: once in ``_process_dino_batched``, once in
    ``process_SAM_to_h5_FastSAM``).
    """

    def __init__(
        self,
        images: list[Path],
        desired_width: int,
        desired_height: int,
        rmin: int,
    ):
        self.images = images
        self.W = desired_width
        self.H = desired_height
        self.rmin = rmin
        h_new = (self.H // 14) * 14
        w_new = (self.W // 14) * 14
        self.crop = tvf.CenterCrop((h_new, w_new))
        self.tf = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        p = self.images[i]
        im = cv2.imread(str(p))[self.rmin :, :, :]
        im_resized = cv2.resize(im, (self.W, self.H))  # BGR, (H, W, 3)
        rgb = cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB)
        dino_tensor = self.crop(self.tf(rgb))
        return p.name, dino_tensor, im_resized


def _collate(batch):
    names = [b[0] for b in batch]
    dino_tensors = torch.stack([b[1] for b in batch])
    bgr_images = [b[2] for b in batch]
    return names, dino_tensors, bgr_images


@dataclass
class _ReservoirStats:
    capacity: int
    fill: int
    seen: int


class _GpuReservoir:
    """Algorithm-R reservoir sampling on a pre-allocated GPU buffer.

    Stores up to ``capacity`` row samples uniformly drawn from a stream of
    unknown length. The buffer is fp16 by default — at 32*1536 = 49152 cols
    and 50k capacity that is ~4.9 GB on GPU (fp32 would be ~9.8 GB).
    """

    def __init__(
        self,
        capacity: int,
        dim: int,
        device,
        dtype: torch.dtype = torch.float16,
    ):
        self.buf = torch.empty((capacity, dim), device=device, dtype=dtype)
        self.capacity = capacity
        self.fill = 0
        self.seen = 0

    def add(self, rows: torch.Tensor) -> None:
        if rows.numel() == 0:
            return
        rows = rows.to(self.buf.dtype)
        n = rows.shape[0]

        if self.fill < self.capacity:
            take = min(self.capacity - self.fill, n)
            self.buf[self.fill : self.fill + take] = rows[:take]
            self.fill += take
            self.seen += take
            rows = rows[take:]
            n -= take
            if n == 0:
                return

        # Batched Algorithm-R: for the i-th remaining row, draw j uniformly
        # from [0, seen+i+1); if j < capacity, replace slot j with this row.
        seen_after = self.seen + torch.arange(1, n + 1, device=rows.device, dtype=torch.int64)
        rand = torch.randint(0, 2**31 - 1, (n,), device=rows.device, dtype=torch.int64)
        j = rand % seen_after
        keep = j < self.capacity
        if keep.any():
            self.buf[j[keep]] = rows[keep]
        self.seen += n

    def tensor(self) -> torch.Tensor:
        return self.buf[: self.fill]

    def stats(self) -> _ReservoirStats:
        return _ReservoirStats(self.capacity, self.fill, self.seen)


class FusedFeaturePcaStage:
    """Fused DINO + FastSAM + per-image SegVLAD + PCA-fit GPU stage.

    Replaces the legacy ``FeatureExtractor → PcaFitter.fit_or_load`` chain
    with a single batched pass that keeps DINO features and FastSAM masks
    on GPU between stages, samples per-image VLADs into a GPU reservoir,
    and writes the H5 caches (dino/masks, optionally vlads) on background
    writer threads so HDF5 compression never blocks GPU compute.

    Outputs are schema-compatible with the legacy ``FeatureStore`` so
    ``VladComputer.compute_all`` works unchanged (and, with the optional
    ``vlads_{W}.h5`` cache, can skip the second SegVLAD pass entirely).
    """

    def __init__(
        self,
        ctx: RuntimeCtx,
        *,
        config: Config,
        c_centers: torch.Tensor,
        ind_matrix: torch.Tensor,
        timer: Optional[Timer] = None,
    ):
        self.ctx = ctx
        self.config = config
        self.timer = timer
        self.legacy_cfg = _legacy_cfg(self.config.desired_width, self.config.desired_height)
        self.c_centers = c_centers
        self.ind_matrix = ind_matrix

        self._dino = None
        self._fastsam = None

    def _load_dino(self):
        if self._dino is None:
            self._dino = func_vpr.loadDINO(
                self.legacy_cfg,
                self.config.dino_model_name,
                self.config.dino_checkpoint,
                device=str(self.ctx.device),
            )
        return self._dino

    def _load_fastsam(self):
        if self._fastsam is None:
            self._fastsam = func_vpr.loadSAM_FastSAM(
                self.config.fastsam_checkpoint,
                self.legacy_cfg,
                device=str(self.ctx.device),
            )
        return self._fastsam

    def _timed(self, name: str):
        if self.timer is None:
            import contextlib

            return contextlib.nullcontext()
        return self.timer.timeit_block(name)

    def run(
        self,
        image_folder: Path,
        images: list[Path],
        output_folder: Path,
        pca_cache_path: Path,
    ) -> tuple[FeatureStore, PcaModel]:
        feature_dir = output_folder
        feature_dir.mkdir(parents=True, exist_ok=True)
        dino_h5 = feature_dir / f"dino_{self.config.desired_width}.h5"
        masks_h5 = feature_dir / f"masks_{self.config.desired_width}.h5"
        vlads_h5 = feature_dir / f"vlads_{self.config.desired_width}.h5" if self.config.cache_all_vlads else None

        full_cache_hit = (
            dino_h5.is_file()
            and masks_h5.is_file()
            and pca_cache_path.is_file()
            and (vlads_h5 is None or vlads_h5.is_file())  # need vlad?
        )
        if full_cache_hit:
            logger.info(f"Fused cache hit: reusing {dino_h5}, {masks_h5}, {pca_cache_path}, vlads={vlads_h5}")
            pca_model = PcaModel.load(pca_cache_path, self.ctx)
            return (
                FeatureStore(dino_h5, masks_h5, vlads_h5),
                pca_model,
            )

        # Any partial cache state means a previous run was interrupted; we
        # rebuild everything (h5py "w" mode truncates so we don't unlink
        # explicitly — keeps us out of the user's other files in this dir).
        # from torch.profiler import profile, record_function, ProfilerActivity

        # 1エポックや数バッチ分だけプロファイリングする
        # with profile(
        # activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # record_shapes=True,
        # profile_memory=True, # メモリの推移も記録
        # with_stack=True
        # ) as prof:
        with self._timed("Fused extraction + PCA fit"):
            self._run_fused(images, dino_h5, masks_h5, vlads_h5, pca_cache_path)
        # prof.export_chrome_trace("run_script_output/trace.json")

        return FeatureStore(dino_h5, masks_h5, vlads_h5), PcaModel.load(pca_cache_path, self.ctx)

    def _run_fused(
        self,
        images: list[Path],
        dino_h5: Path,
        masks_h5: Path,
        vlads_h5: Optional[Path],
        pca_cache_path: Path,
    ) -> None:
        dino = self._load_dino()
        fastsam = self._load_fastsam()

        ds = _BatchDataset(
            images,
            self.config.desired_width,
            self.config.desired_height,
            self.legacy_cfg["rmin"],
        )

        batch_size = self.config.runtime.fused_batch_size
        if batch_size == 0:
            from revisit_anything.autobatch import estimate_fused_batch_size

            batch_size = int(np.floor(0.9*estimate_fused_batch_size(
                dino,
                fastsam,
                images,
                self.legacy_cfg,
                self.ctx,
            )))
            logger.info(f"Using auto-detected fused_batch_size={batch_size}")

        loader = DataLoader(
            ds,
            batch_size=batch_size if batch_size > 0 else 1,
            num_workers=self.config.runtime.dataloader_num_workers,
            shuffle=False,
            collate_fn=_collate,
            pin_memory=(self.ctx.device.type == "cuda"),
        )

        h_new = (self.config.desired_height // 14) * 14
        w_new = (self.config.desired_width // 14) * 14
        h_r = h_new // 14
        w_r = w_new // 14
        vlad_dim = self.c_centers.shape[0] * self.config.vlad_config.desc_dim

        reservoir = _GpuReservoir(
            capacity=self.config.pca_config.max_segments,
            dim=vlad_dim,
            device=self.ctx.device,
            dtype=torch.float16,
        )

        dino_w = AsyncH5Writer(dino_h5, dino_write_fn, max_queue=self.config.runtime.h5_writer_queue)
        masks_w = AsyncH5Writer(masks_h5, masks_write_fn, max_queue=self.config.runtime.h5_writer_queue)
        vlads_w: Optional[AsyncH5Writer] = (
            AsyncH5Writer(vlads_h5, vlads_write_fn, max_queue=self.config.runtime.h5_writer_queue)
            if vlads_h5 is not None
            else None
        )

        # Pre-bind loop locals so the post-loop cleanup never trips on
        # NameError even if the loader yields zero batches.
        batch = feat = feat_cpu = None
        mask_gpu = adj_mat = vlad = None
        fastsam_masks = None

        try:
            with torch.no_grad(), self.ctx.autocast():
                global_idx = 0
                for names, dino_tensors, bgr_images in tqdm(loader, desc="Fused DINO+FastSAM+VLAD"):
                    batch = dino_tensors.to(self.ctx.device, non_blocking=True)
                    feat = dino(batch)  # [B, P, D]
                    B = batch.shape[0]
                    feat = feat.reshape(B, h_r, w_r, self.config.vlad_config.desc_dim).permute(0, 3, 1, 2)
                    feat = F.normalize(feat, dim=1)

                    fastsam_masks = func_vpr.process_batch_FastSAM(
                        self.legacy_cfg,
                        bgr_images,
                        fastsam,
                        names,
                        device=str(self.ctx.device),
                    )

                    feat_cpu = feat.detach().cpu().numpy()

                    for i, name in enumerate(names):
                        seg_list = [m["segmentation"] for m in fastsam_masks[i]]
                        if len(seg_list) == 0:
                            # process_batch_FastSAM already injects a fallback
                            # mask pair when ann is empty, so this branch is
                            # unreachable. Guard anyway.
                            global_idx += 1
                            continue

                        mask_np = np.stack(seg_list, axis=0)
                        mask_gpu = torch.from_numpy(mask_np).to(self.ctx.device)
                        adj_mat = _build_adj_gpu(mask_gpu, self.config.pca_config.order if self.config.pca_fit else 0)
                        vlad = seg_vlad_gpu_from_tensors(
                            feat[i],
                            mask_gpu,
                            self.c_centers,
                            self.ind_matrix,
                            desired_height=self.config.desired_height,
                            desired_width=self.config.desired_width,
                            desc_dim=self.config.vlad_config.desc_dim,
                            adj_mat=adj_mat,
                            device=str(self.ctx.device),
                        )
                        reservoir.add(vlad)

                        dino_w.put((name, feat_cpu[i : i + 1]))
                        masks_w.put((name, seg_list))
                        if vlads_w is not None:
                            vlad_cpu = vlad.detach().to(torch.float16).cpu().numpy()
                            vlads_w.put((name, vlad_cpu))

                        global_idx += 1

            batch = feat = feat_cpu = None
            mask_gpu = adj_mat = vlad = None
            fastsam_masks = None
            dino = fastsam = None
            loader = None  # type: ignore[assignment]
            self._clear_memory()

            stats = reservoir.stats()
            logger.info(f"Reservoir: fill={stats.fill}/{stats.capacity} seen={stats.seen}")

            pca = PcaFitter(
                self.ctx,
                n_components=self.config.pca_config.lower_dim,
                whiten=self.config.pca_config.whiten,
                svd_solver=self.config.pca_config.svd_solver,
                max_segments=self.config.pca_config.max_segments,
            ).fit_from_buffer(reservoir.tensor())
            pca.save(pca_cache_path)
            logger.info(f"PCA model saved to {pca_cache_path}")
        except Exception as e:
            logger.exception(f"Error during fused stage: {e}")
            raise
        finally:
            dino_w.close()
            masks_w.close()
            if vlads_w is not None:
                vlads_w.close()

    def close(self) -> None:
        self._dino = None
        self._fastsam = None
        if self.ctx.device.type == "cuda":
            torch.cuda.empty_cache()

    def _clear_memory(self) -> None:
        before = torch.cuda.memory_allocated() / 1024**3 if self.ctx.device.type == "cuda" else 0.0
        self._dino = None
        self._fastsam = None
        # Loader holds CPU memory only, but breaking the reference lets
        # the (forked) worker subprocesses exit promptly.
        gc.collect()
        if self.ctx.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            after = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU before/after cleanup: {before:.2f}/{after:.2f} GB; ")
