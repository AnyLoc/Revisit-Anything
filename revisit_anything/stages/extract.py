from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional

import cv2
import h5py
import torch
import torchvision.transforms as tvf
from loguru import logger
from tqdm import tqdm

import revisit_anything.utils as func_vpr
from revisit_anything.feature_store import FeatureStore
from revisit_anything.runtime import RuntimeCtx
from revisit_anything.timer import Timer


def _chunks(seq: Iterable, n: int) -> Iterator[list]:
    it = iter(seq)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


# The legacy func_vpr loaders expect a dict with these flags. The original
# code copy-pasted this dict three times with identical values; we centralize
# it here. Only the spatial dims change between callers.
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


class FeatureExtractor:
    """Stage 1: run DINOv2 + a mask segmenter (SAM or FastSAM) and cache to H5.

    Models are loaded once in ``__init__`` and live until ``close()`` is called.
    ``extract()`` produces a ``FeatureStore`` (or just opens the existing
    cached H5 files if both already exist).
    """

    def __init__(
        self,
        ctx: RuntimeCtx,
        *,
        dino_model: str,
        segmenter: Literal["sam", "fastsam"],
        desired_width: int,
        desired_height: int,
        dino_checkpoint: Optional[str] = None,
        sam_checkpoint: Optional[str] = None,
        fastsam_checkpoint: Optional[str] = None,
        timer: Optional[Timer] = None,
        dino_batch_size: int = 8,
    ):
        if segmenter == "sam" and not sam_checkpoint:
            raise ValueError("segmenter='sam' requires sam_checkpoint")
        if segmenter == "fastsam" and not fastsam_checkpoint:
            raise ValueError("segmenter='fastsam' requires fastsam_checkpoint")
        if segmenter not in ("sam", "fastsam"):
            raise ValueError(f"Unknown segmenter: {segmenter!r}")

        self.ctx = ctx
        self.dino_model_name = dino_model
        self.segmenter = segmenter
        self.dino_checkpoint = dino_checkpoint
        self.sam_checkpoint = sam_checkpoint
        self.fastsam_checkpoint = fastsam_checkpoint
        self.desired_width = desired_width
        self.desired_height = desired_height
        self.timer = timer
        self.dino_batch_size = dino_batch_size
        self.legacy_cfg = _legacy_cfg(desired_width, desired_height)

        self._dino = None
        self._segmenter_model = None

    def _load_dino(self):
        if self._dino is None:
            self._dino = func_vpr.loadDINO(
                self.legacy_cfg, self.dino_model_name, self.dino_checkpoint, device=str(self.ctx.device)
            )
        return self._dino

    def _load_segmenter(self):
        if self._segmenter_model is None:
            if self.segmenter == "sam":
                self._segmenter_model = func_vpr.loadSAM(
                    self.sam_checkpoint, self.legacy_cfg, device=str(self.ctx.device)
                )
            else:
                self._segmenter_model = func_vpr.loadSAM_FastSAM(
                    self.fastsam_checkpoint, self.legacy_cfg, device=str(self.ctx.device)
                )
        return self._segmenter_model

    def _timed(self, name: str):
        if self.timer is None:
            import contextlib
            return contextlib.nullcontext()
        return self.timer.timeit_block(name)

    def extract(
        self,
        image_folder: Path,
        images: list[Path],
        output_folder: Path,
    ) -> FeatureStore:
        feature_dir = output_folder# / "image_feature_dicts"
        feature_dir.mkdir(parents=True, exist_ok=True)
        dino_h5 = feature_dir / f"dino_{self.desired_width}.h5"
        masks_h5 = feature_dir / f"masks_{self.desired_width}.h5"

        if dino_h5.is_file() and masks_h5.is_file():
            logger.info(f"Feature H5 cache hit: {dino_h5} / {masks_h5}")
            return FeatureStore(dino_h5, masks_h5)

        if not dino_h5.is_file():
            logger.info(f"Starting DINO extraction (batched, B={self.dino_batch_size})")
            with self._timed("DINOfeature extraction"), self.ctx.autocast():
                self._process_dino_batched(dino_h5, images)
                # func_vpr.process_dino_ft_to_h5(dino_h5, self.legacy_cfg, images, self._load_dino(), device=str(self.ctx.device))

        if not masks_h5.is_file():
            logger.info(f"Starting {self.segmenter.upper()} extraction")
            with self._timed(f"{self.segmenter.upper()} feature extraction"):
                if self.segmenter == "sam":
                    func_vpr.process_SAM_to_h5(
                        masks_h5, self.legacy_cfg, images, self._load_segmenter(),
                        device=str(self.ctx.device), dataDir=str(image_folder),
                    )
                else:
                    func_vpr.process_SAM_to_h5_FastSAM(
                        masks_h5, self.legacy_cfg, images, self._load_segmenter(),
                        device=str(self.ctx.device), dataDir=str(image_folder),
                    )

        return FeatureStore(dino_h5, masks_h5)

    def _process_dino_batched(self, dino_h5: Path, images: list[Path]) -> None:
        """Batched replacement for ``func_vpr.process_dino_ft_to_h5``.

        The legacy helper runs one image per DINOv2 forward pass, leaving the
        GPU mostly idle on ViT-g/14. Here we stack ``dino_batch_size`` images
        into ``[B, C, H, W]`` and write the resulting features to H5 with the
        same per-image group/dataset schema (so downstream readers are unchanged).
        """
        rmin = self.legacy_cfg["rmin"]
        W, H = self.desired_width, self.desired_height
        h_new = (H // 14) * 14
        w_new = (W // 14) * 14
        h_r, w_r = h_new // 14, w_new // 14

        base_tf = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        crop = tvf.CenterCrop((h_new, w_new))
        dino = self._load_dino()
        device = str(self.ctx.device)

        with h5py.File(dino_h5, "w") as f:
            for chunk in tqdm(
                _chunks(images, self.dino_batch_size),
                total=(len(images) + self.dino_batch_size - 1) // self.dino_batch_size,
            ):
                tensors, names = [], []
                for p in chunk:
                    im = cv2.imread(str(p))[rmin:, :, :]
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, (W, H))
                    tensors.append(base_tf(im))
                    names.append(p.name)
                batch = crop(torch.stack(tensors)).to(device)
                feat = dino(batch)  # [B, num_patches, desc_dim]
                feat = feat.reshape(batch.shape[0], h_r, w_r, -1).permute(0, 3, 1, 2)
                feat = torch.nn.functional.normalize(feat, dim=1)
                feat_np = feat.detach().cpu().numpy()
                for i, name in enumerate(names):
                    grp = f.create_group(name)
                    grp.create_dataset("ift_dino", data=feat_np[i:i+1], chunks=True)

    def close(self) -> None:
        """Release model handles and free CUDA memory."""
        self._dino = None
        self._segmenter_model = None
        if self.ctx.device.type == "cuda":
            torch.cuda.empty_cache()
