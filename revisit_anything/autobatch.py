"""Auto-detect a safe ``fused_batch_size`` for the fused DINO+FastSAM stage.

Profiles the DINO and FastSAM forwards at a few candidate batch sizes,
fits ``peak_memory(B) ≈ a*B + b`` and solves for the largest ``B`` whose
predicted peak stays under ``free * fraction`` of GPU memory.

Patterned after ``submodules/FastSAM/ultralytics/yolo/utils/autobatch.py``
but specialised to the two-model pipeline used in
``revisit_anything/stages/fused.py``. The per-image VLAD/mask aggregation
loop is intentionally excluded from the probe because it doesn't scale
with batch size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger

import revisit_anything.utils as func_vpr
from revisit_anything.runtime import RuntimeCtx


_GB = 1 << 30


def estimate_fused_batch_size(
    dino,
    fastsam,
    images: list[Path],
    legacy_cfg: dict,
    ctx: RuntimeCtx,
    *,
    fraction: float = 0.70,
    probe_batches: tuple[int, ...] = (1, 2, 4),
    bounds: tuple[int, int] = (1, 64),
    fallback: int = 8,
) -> int:
    """Return a safe ``fused_batch_size`` for the current GPU.

    Args:
        dino: Loaded DINO model (callable on ``[B, 3, H, W]`` tensors).
        fastsam: Loaded FastSAM model accepted by
            ``func_vpr.process_batch_FastSAM``.
        images: Real images to draw probe batches from. At least
            ``max(probe_batches)`` paths must be available.
        legacy_cfg: The same dict passed to ``process_batch_FastSAM`` in
            production (carries ``desired_width``/``desired_height``/etc.).
        ctx: Runtime context (used for device + autocast).
        fraction: Target peak-memory utilisation of free GPU. 0.70 leaves
            headroom for the per-image VLAD/mask loop which the probe
            doesn't cover.
        probe_batches: Candidate batch sizes to measure. Each is run once.
        bounds: ``(min, max)`` clamp for the final answer.
        fallback: Returned when CUDA is unavailable, when fewer than two
            probes succeed, or on any unexpected error.
    """
    if ctx.device.type != "cuda":
        logger.info(
            f"autobatch: device={ctx.device}, returning fallback={fallback}"
        )
        return fallback

    max_probe = max(probe_batches)
    if len(images) < max_probe:
        logger.warning(
            f"autobatch: only {len(images)} probe images available "
            f"(need {max_probe}), returning fallback={fallback}"
        )
        return fallback

    device = ctx.device
    try:
        # Memory snapshot before profiling.
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory / _GB
        reserved = torch.cuda.memory_reserved(device) / _GB
        allocated = torch.cuda.memory_allocated(device) / _GB
        free = total - max(reserved, allocated)
        logger.info(
            f"autobatch: {device} ({props.name}) "
            f"total={total:.2f}G reserved={reserved:.2f}G "
            f"allocated={allocated:.2f}G free={free:.2f}G"
        )

        # Build probe tensors from the real images so DINO/FastSAM see
        # realistic content (FastSAM peak memory depends on the number of
        # detected masks which depends on the image).
        probe_paths = images[:max_probe]
        from revisit_anything.stages.fused import _BatchDataset, _collate

        ds = _BatchDataset(
            probe_paths,
            legacy_cfg["desired_width"],
            legacy_cfg["desired_height"],
            legacy_cfg["rmin"],
        )
        probe_batch = _collate([ds[i] for i in range(max_probe)])
        all_names, all_dino, all_bgr = probe_batch

        peaks: list[tuple[int, float]] = []
        oom_at: Optional[int] = None

        with torch.no_grad(), ctx.autocast():
            for b in probe_batches:
                names = all_names[:b]
                dino_tensors = all_dino[:b].to(device, non_blocking=True)
                bgr_images = all_bgr[:b]
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    _ = dino(dino_tensors)
                    _ = func_vpr.process_batch_FastSAM(
                        legacy_cfg,
                        bgr_images,
                        fastsam,
                        names,
                        device=str(device),
                    )
                    torch.cuda.synchronize(device)
                    peak = torch.cuda.max_memory_allocated(device) / _GB
                    peaks.append((b, peak))
                    logger.info(
                        f"autobatch: B={b} peak={peak:.2f}G"
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"autobatch: OOM at B={b}, stopping probe")
                    oom_at = b
                    torch.cuda.empty_cache()
                    break
                finally:
                    del dino_tensors

        if len(peaks) < 2:
            logger.warning(
                f"autobatch: only {len(peaks)} successful probe(s), "
                f"returning fallback={fallback}"
            )
            return fallback

        xs = np.array([b for b, _ in peaks], dtype=np.float64)
        ys = np.array([m for _, m in peaks], dtype=np.float64)
        slope, intercept = np.polyfit(xs, ys, deg=1)
        if slope <= 0:
            logger.warning(
                f"autobatch: non-positive slope ({slope:.3f}G/B), "
                f"returning fallback={fallback}"
            )
            return fallback

        target = free * fraction
        b_star = int((target - intercept) / slope)

        # Clamp below the OOM point if we hit one.
        if oom_at is not None and b_star >= oom_at:
            # Largest probe that succeeded.
            b_star = peaks[-1][0]

        b_star = max(bounds[0], min(bounds[1], b_star))
        predicted = slope * b_star + intercept
        logger.info(
            f"autobatch: chose B={b_star} "
            f"(predicted peak={predicted:.2f}G, target={target:.2f}G, "
            f"fit: {slope:.3f}G*B+{intercept:.3f}G)"
        )

        # Release any cached blocks the probe left behind so the actual
        # pipeline starts from a clean slate.
        torch.cuda.empty_cache()
        return b_star

    except Exception as e:
        logger.warning(
            f"autobatch: unexpected error ({type(e).__name__}: {e}), "
            f"returning fallback={fallback}"
        )
        return fallback
