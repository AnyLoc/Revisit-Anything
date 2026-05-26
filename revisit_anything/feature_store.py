from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
from loguru import logger
from tqdm import tqdm


class FeatureStore:
    """Open handle bundle for cached DINO descriptors + segmentation masks.

    Owns two ``h5py.File`` handles. Use as a context manager so they are
    closed at pipeline end. The raw ``.dino_h5`` handle is exposed because
    ``func_vpr.seg_vlad_gpu_single`` consumes the open H5 directly.

    ``vlads_h5_path`` is optional and, when set, points to a pre-computed
    per-image VLAD cache produced by :class:`FusedFeaturePcaStage`. We do
    not open it here because :meth:`VladComputer._compute_all_from_cache`
    opens it for the duration of its single pass.
    """

    def __init__(
        self,
        dino_h5_path: Path,
        masks_h5_path: Path,
        vlads_h5_path: "Path | None" = None,
    ):
        self.dino_h5_path = Path(dino_h5_path)
        self.masks_h5_path = Path(masks_h5_path)
        self.vlads_h5_path = Path(vlads_h5_path) if vlads_h5_path is not None else None
        self.dino_h5 = h5py.File(str(self.dino_h5_path), "r")
        self.masks_h5 = h5py.File(str(self.masks_h5_path), "r")

    def __enter__(self) -> "FeatureStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.dino_h5.close()
        finally:
            self.masks_h5.close()

    def count_total_segments(self, image_names: Iterable[str]) -> int:
        """Sum number of stored mask entries across the given images."""
        total = 0
        for name in tqdm(image_names, desc="Counting masks"):
            grp = self.masks_h5.get(f"{name}/masks/")
            if grp is not None:
                total += len(grp.keys())
        logger.info(f"Total mask count: {total}")
        return total
