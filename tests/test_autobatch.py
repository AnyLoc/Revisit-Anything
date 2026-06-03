"""Unit tests for ``revisit_anything.autobatch.estimate_fused_batch_size``.

Designed to run inside the ``sfm_project`` docker image where the submodule
is mounted at ``/root/polish-pipeline/sfm_project/Revisit-Anything``. The
tests exercise the helper without depending on real DINO/FastSAM weights:
the CPU path short-circuits before any model call, and the CUDA paths
inject stub callables whose memory footprint we control with explicit
allocations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from revisit_anything.autobatch import estimate_fused_batch_size
from revisit_anything.runtime import RuntimeCtx


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _ctx(device: str) -> RuntimeCtx:
    return RuntimeCtx(
        device=torch.device(device),
        dtype=torch.float32,
        faiss_backend="cpu",
        pca_backend="torch_svd",
    )


def _legacy_cfg(w: int = 56, h: int = 28) -> dict:
    return {
        "desired_width": w,
        "desired_height": h,
        "rmin": 0,
        "resize": True,
    }


def _write_dummy_images(tmp_path: Path, n: int, w: int, h: int) -> list[Path]:
    """``_BatchDataset`` reads with cv2.imread, so write real PNGs."""
    import cv2

    paths: list[Path] = []
    rs = np.random.RandomState(0)
    for i in range(n):
        p = tmp_path / f"img_{i:03d}.png"
        img = (rs.rand(h, w, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


def test_cpu_returns_fallback() -> None:
    """On CPU the function must short-circuit to ``fallback`` without
    touching the model callables."""

    def boom(*a, **kw):
        raise AssertionError("model must not be called on CPU")

    result = estimate_fused_batch_size(
        boom,
        boom,
        [],
        _legacy_cfg(),
        _ctx("cpu"),
        fallback=11,
    )
    assert result == 11


def test_cpu_with_unused_images(tmp_path: Path) -> None:
    """Same as above but with real images present; behaviour unchanged."""
    images = _write_dummy_images(tmp_path, 4, 56, 28)
    result = estimate_fused_batch_size(
        lambda x: x,
        lambda *a, **kw: [],
        images,
        _legacy_cfg(),
        _ctx("cpu"),
        fallback=7,
    )
    assert result == 7


@requires_cuda
def test_insufficient_images_returns_fallback(tmp_path: Path) -> None:
    """When fewer probe images are available than ``max(probe_batches)``,
    we cannot run the probe; return ``fallback``."""
    images = _write_dummy_images(tmp_path, 2, 56, 28)  # need 4 for default
    result = estimate_fused_batch_size(
        lambda x: x,
        lambda *a, **kw: [],
        images,
        _legacy_cfg(),
        _ctx("cuda"),
        fallback=5,
    )
    assert result == 5


@requires_cuda
def test_linear_fit_picks_target(tmp_path: Path, monkeypatch) -> None:
    """With a stub that allocates ``per_image * B`` bytes the linear fit
    must solve to roughly ``free * fraction / per_image_GB``."""
    images = _write_dummy_images(tmp_path, 4, 56, 28)

    # Use a small constant-size dino+fastsam stub so the *only* batch-size
    # signal comes from the input tensor that the probe ships to the GPU.
    # _BatchDataset normalises images into fp32 tensors of shape
    # [B, 3, h_new, w_new] where (h_new, w_new) is (14, 56) for this cfg
    # → per-image footprint = 3*14*56*4 bytes = 9408 B (≈ 9 KB).
    # That's too tiny to see in the GB-fit; instead we make the stub
    # itself allocate a sizeable per-image scratch tensor so the slope is
    # measurable.
    PER_IMAGE_FLOATS = 16 * 1024 * 1024  # 64 MB per image (fp32)

    captured_inputs: list[int] = []

    def dino_stub(x: torch.Tensor) -> torch.Tensor:
        # Allocate a scratch buffer proportional to B and keep a handle
        # so it survives until reset_peak_memory_stats picks it up.
        b = x.shape[0]
        captured_inputs.append(b)
        scratch = torch.empty((b, PER_IMAGE_FLOATS), device=x.device, dtype=torch.float32)
        # Touch it so the allocation actually happens.
        scratch.fill_(0)
        return scratch

    def fastsam_stub(cfg, bgr_images, model, imnames, device):
        # Return the empty mask fallback shape the real helper uses.
        h, w = cfg["desired_height"], cfg["desired_width"]
        return [[{"segmentation": np.zeros((h, w), dtype=bool)}] for _ in imnames]

    # Patch process_batch_FastSAM in the autobatch module.
    import revisit_anything.autobatch as ab

    monkeypatch.setattr(ab.func_vpr, "process_batch_FastSAM", fastsam_stub)

    # Force a known "free" memory so the picked B is predictable.
    # 64MB per image, target = free*fraction → B ≈ target/64MB.
    # We can't override get_device_properties, but we can choose
    # fraction such that the answer falls inside `bounds`.
    result = estimate_fused_batch_size(
        dino_stub,
        object(),
        images,
        _legacy_cfg(),
        _ctx("cuda"),
        fraction=0.05,  # keep B small so we don't OOM the test box
        probe_batches=(1, 2),
        bounds=(1, 64),
        fallback=99,
    )
    # Both probes must have run, and the chosen B must be a real fit, not
    # the fallback.
    assert captured_inputs[:2] == [1, 2]
    assert result != 99
    assert 1 <= result <= 64


@requires_cuda
def test_oom_in_probe_falls_back_to_largest_succeeding(tmp_path: Path, monkeypatch) -> None:
    """If the largest probe OOMs, the chosen B must not exceed the largest
    probe that succeeded."""
    images = _write_dummy_images(tmp_path, 4, 56, 28)

    def dino_stub(x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        if b >= 4:
            raise torch.cuda.OutOfMemoryError("simulated")
        return torch.empty((b, 16 * 1024 * 1024), device=x.device, dtype=torch.float32).fill_(0)

    def fastsam_stub(cfg, bgr_images, model, imnames, device):
        h, w = cfg["desired_height"], cfg["desired_width"]
        return [[{"segmentation": np.zeros((h, w), dtype=bool)}] for _ in imnames]

    import revisit_anything.autobatch as ab

    monkeypatch.setattr(ab.func_vpr, "process_batch_FastSAM", fastsam_stub)

    result = estimate_fused_batch_size(
        dino_stub,
        object(),
        images,
        _legacy_cfg(),
        _ctx("cuda"),
        fraction=0.95,  # try to push hard so unclamped B>=4
        probe_batches=(1, 2, 4),
        bounds=(1, 64),
        fallback=99,
    )
    # Largest probe to succeed was 2; result must respect that cap.
    assert result <= 2
    assert result != 99


@requires_cuda
def test_too_few_successful_probes_returns_fallback(tmp_path: Path, monkeypatch) -> None:
    """If only one probe succeeds we can't fit a line; fall back."""
    images = _write_dummy_images(tmp_path, 4, 56, 28)

    def dino_stub(x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        if b >= 2:
            raise torch.cuda.OutOfMemoryError("simulated")
        return torch.empty((b, 1024 * 1024), device=x.device, dtype=torch.float32).fill_(0)

    def fastsam_stub(cfg, bgr_images, model, imnames, device):
        h, w = cfg["desired_height"], cfg["desired_width"]
        return [[{"segmentation": np.zeros((h, w), dtype=bool)}] for _ in imnames]

    import revisit_anything.autobatch as ab

    monkeypatch.setattr(ab.func_vpr, "process_batch_FastSAM", fastsam_stub)

    result = estimate_fused_batch_size(
        dino_stub,
        object(),
        images,
        _legacy_cfg(),
        _ctx("cuda"),
        probe_batches=(1, 2, 4),
        fallback=13,
    )
    assert result == 13
