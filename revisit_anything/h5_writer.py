from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any, Callable, Optional

import h5py
import numpy as np
from loguru import logger


_STOP = object()


class AsyncH5Writer:
    """Single-threaded h5py writer with a bounded queue.

    h5py is not thread-safe per file, so each writer owns its own thread and
    its own ``h5py.File`` handle. The bounded queue applies back-pressure to
    the producer when LZF compression is slower than GPU compute, preventing
    unbounded CPU memory growth.

    ``write_fn`` is called as ``write_fn(open_h5_file, item)`` for every
    item pushed via :meth:`put`. The producer is responsible for moving GPU
    tensors to CPU (numpy/torch) before submission so the writer thread does
    not have to interact with CUDA.
    """

    def __init__(
        self,
        path: Path,
        write_fn: Callable[[h5py.File, Any], None],
        max_queue: int = 16,
    ):
        self.path = Path(path)
        self._write_fn = write_fn
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._error: Optional[BaseException] = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name=f"h5w[{self.path.name}]", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            with h5py.File(str(self.path), "w") as f:
                while True:
                    item = self._q.get()
                    try:
                        if item is _STOP:
                            return
                        self._write_fn(f, item)
                    finally:
                        self._q.task_done()
        except BaseException as e:
            self._error = e
            logger.exception(f"AsyncH5Writer thread for {self.path} crashed")

    def put(self, item: Any) -> None:
        if self._error is not None:
            raise RuntimeError(
                f"AsyncH5Writer for {self.path} has errored"
            ) from self._error
        if self._closed:
            raise RuntimeError(
                f"AsyncH5Writer for {self.path} is already closed"
            )
        self._q.put(item)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._q.put(_STOP)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError(
                f"AsyncH5Writer for {self.path} crashed"
            ) from self._error

    def __enter__(self) -> "AsyncH5Writer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def dino_write_fn(f: h5py.File, item: tuple[str, np.ndarray]) -> None:
    """Schema-matching writer for ``dino_{W}.h5`` (one group per image).

    Item is ``(image_name, feat_np)`` where ``feat_np`` has shape
    ``[1, D, h_r, w_r]`` matching the legacy ``FeatureExtractor`` output
    (``extract.py:201``).
    """
    name, feat = item
    grp = f.create_group(name)
    grp.create_dataset("ift_dino", data=feat, chunks=True, compression="gzip")


def masks_write_fn(f: h5py.File, item: tuple[str, list[np.ndarray]]) -> None:
    """Schema-matching writer for ``masks_{W}.h5``.

    Item is ``(image_name, [seg_0, seg_1, ...])`` where each entry is a
    boolean ``(H, W)`` array. The on-disk schema is identical to the legacy
    ``process_SAM_to_h5_FastSAM`` output (``utils.py:684-689``).
    """
    name, masks_seg = item
    grp = f.create_group(name)
    masks_grp = grp.create_group("masks")
    for j, seg in enumerate(masks_seg):
        masks_grp.create_dataset(
            f"{j}/segmentation", data=seg, compression="lzf", chunks=True
        )


def vlads_write_fn(
    f: h5py.File, item: tuple[str, np.ndarray, np.ndarray]
) -> None:
    """Writer for the optional ``vlads_{W}.h5`` cache.

    Item is ``(image_name, vlad_np, im_inds_np)``. ``vlad_np`` is the raw
    pre-PCA VLAD vector with shape ``[n_seg, vlad_dim]`` (we store fp16 to
    keep disk footprint bounded; PCA fit will up-cast at read time).
    """
    name, vlad, im_inds = item
    grp = f.create_group(name)
    grp.create_dataset("vlad", data=vlad, chunks=True, compression="gzip")
    grp.create_dataset("im_inds", data=im_inds, chunks=True, compression="gzip")
