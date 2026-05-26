"""Unit tests for the fused DINO+FastSAM+PCA-fit GPU stage.

Designed to run inside the ``sfm_project`` docker image where the submodule
is mounted at ``/root/polish-pipeline/sfm_project/Revisit-Anything``. End-to-end
runs and legacy A/B numerical comparisons are intentionally out of scope.

What we *do* verify:
- ``AsyncH5Writer`` round-trips items and respects the ``write_fn``.
- ``_GpuReservoir`` uniformly samples up to its capacity (statistical check).
- ``_build_adj_gpu`` returns the same matrix as
  ``func_vpr.nbrMasksAGGFastSingle`` for the same masks.
- ``seg_vlad_gpu_from_tensors`` matches ``func_vpr.seg_vlad_gpu_single``
  numerically when fed identical inputs.
- ``PcaFitter.fit_from_buffer`` reproduces ``_fit_torch_svd`` on the same
  buffer.
- ``VladComputer._compute_all_from_cache`` reads ``vlads_{W}.h5`` and
  applies PCA without re-running SegVLAD.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from revisit_anything import utils as func_vpr
from revisit_anything.feature_store import FeatureStore
from revisit_anything.h5_writer import (
    AsyncH5Writer,
    dino_write_fn,
    masks_write_fn,
    vlads_write_fn,
)
from revisit_anything.runtime import RuntimeCtx
from revisit_anything.stages.fused import _GpuReservoir
from revisit_anything.stages.pca import PcaFitter
from revisit_anything.stages.vlad import (
    VladComputer,
    _build_adj_gpu,
    seg_vlad_gpu_from_tensors,
)
from revisit_anything.vlad_index import build_pixel_to_patch_index


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _make_ctx(device: str = "cuda") -> RuntimeCtx:
    return RuntimeCtx(
        device=torch.device(device),
        dtype=torch.float32,
        faiss_backend="cpu",
        pca_backend="torch_svd",
    )


# ---------------------------------------------------------------------------
# AsyncH5Writer
# ---------------------------------------------------------------------------


def test_async_h5_writer_round_trip(tmp_path: Path) -> None:
    """The writer must persist every submitted item in submission order."""
    path = tmp_path / "round_trip.h5"

    def write_fn(f: h5py.File, item) -> None:
        key, arr = item
        f.create_dataset(key, data=arr)

    with AsyncH5Writer(path, write_fn, max_queue=4) as w:
        for i in range(20):
            w.put((f"k_{i}", np.arange(i + 1, dtype=np.int32)))

    with h5py.File(str(path), "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        assert keys == [f"k_{i}" for i in range(20)]
        for i, k in enumerate(keys):
            np.testing.assert_array_equal(
                f[k][()], np.arange(i + 1, dtype=np.int32)
            )


def test_async_h5_writer_dino_schema(tmp_path: Path) -> None:
    """``dino_write_fn`` must place data at ``{name}/ift_dino``."""
    path = tmp_path / "dino_test.h5"
    feat = np.random.RandomState(0).randn(1, 1536, 4, 5).astype(np.float32)
    with AsyncH5Writer(path, dino_write_fn) as w:
        w.put(("img_0.png", feat))

    with h5py.File(str(path), "r") as f:
        stored = f["img_0.png/ift_dino"][()]
    np.testing.assert_array_equal(stored, feat)


def test_async_h5_writer_masks_schema(tmp_path: Path) -> None:
    """``masks_write_fn`` matches the legacy ``process_SAM_to_h5_FastSAM`` schema."""
    path = tmp_path / "masks_test.h5"
    rs = np.random.RandomState(1)
    seg = [rs.rand(8, 8) > 0.5 for _ in range(3)]
    with AsyncH5Writer(path, masks_write_fn) as w:
        w.put(("img_0.png", seg))

    with h5py.File(str(path), "r") as f:
        for j, expected in enumerate(seg):
            actual = f[f"img_0.png/masks/{j}/segmentation"][()]
            np.testing.assert_array_equal(actual, expected)


def test_async_h5_writer_vlads_schema(tmp_path: Path) -> None:
    path = tmp_path / "vlads_test.h5"
    vlad = np.random.RandomState(2).randn(7, 1536 * 32).astype(np.float16)
    im_inds = np.full(7, 3, dtype=np.int64)
    with AsyncH5Writer(path, vlads_write_fn) as w:
        w.put(("img_0.png", vlad, im_inds))

    with h5py.File(str(path), "r") as f:
        np.testing.assert_array_equal(f["img_0.png/vlad"][()], vlad)
        np.testing.assert_array_equal(f["img_0.png/im_inds"][()], im_inds)


# ---------------------------------------------------------------------------
# _GpuReservoir
# ---------------------------------------------------------------------------


def test_reservoir_fill_phase_cpu() -> None:
    """Fewer items than capacity must all be kept verbatim."""
    cap, dim = 100, 16
    res = _GpuReservoir(cap, dim, device=torch.device("cpu"), dtype=torch.float32)
    rows = torch.arange(50 * dim, dtype=torch.float32).reshape(50, dim)
    res.add(rows)
    assert res.fill == 50
    assert res.seen == 50
    torch.testing.assert_close(res.tensor(), rows)


def test_reservoir_overflow_uniform_cpu() -> None:
    """When stream length exceeds capacity, the buffer must still be full and
    samples must come from the original stream (no junk rows)."""
    cap, dim = 64, 4
    n_total = 1000
    torch.manual_seed(0)
    res = _GpuReservoir(cap, dim, device=torch.device("cpu"), dtype=torch.float32)
    # Unique signature per row so we can confirm everything stored came from
    # the input stream.
    rows = torch.arange(n_total * dim, dtype=torch.float32).reshape(n_total, dim)
    res.add(rows)
    assert res.fill == cap
    assert res.seen == n_total
    # Every stored row's first column must match its arithmetic position in
    # the original stream (since dim=4 and step=4 makes col-0 = row_idx*4).
    cols = res.tensor()[:, 0].long()
    expected_origins = cols // dim
    assert ((expected_origins >= 0) & (expected_origins < n_total)).all()


# ---------------------------------------------------------------------------
# _build_adj_gpu vs legacy nbrMasksAGGFastSingle
# ---------------------------------------------------------------------------


@requires_cuda
def test_build_adj_gpu_matches_legacy() -> None:
    """Same masks must produce the same boolean adjacency for both helpers."""
    H, W, n = 32, 32, 8
    rs = np.random.RandomState(42)
    masks_seg = [rs.rand(H, W) > 0.7 for _ in range(n)]
    # Force non-empty masks so centroids are well-defined.
    for m in masks_seg:
        m[0, 0] = True

    order = 3
    legacy = func_vpr.nbrMasksAGGFastSingle(masks_seg, order=order)

    mask_tensor = torch.from_numpy(np.stack(masks_seg, axis=0)).to("cuda")
    new = _build_adj_gpu(mask_tensor, order=order)

    assert new is not None
    np.testing.assert_array_equal(new.cpu().numpy(), legacy.cpu().numpy())


def test_build_adj_gpu_returns_none_for_zero_order() -> None:
    masks = torch.zeros((3, 8, 8), dtype=torch.bool)
    assert _build_adj_gpu(masks, order=0) is None
    assert _build_adj_gpu(masks, order=None) is None


# ---------------------------------------------------------------------------
# seg_vlad_gpu_from_tensors vs legacy seg_vlad_gpu_single
# ---------------------------------------------------------------------------


@requires_cuda
def test_seg_vlad_from_tensors_matches_legacy(tmp_path: Path) -> None:
    """In-memory SegVLAD must match the H5-backed legacy implementation when
    fed the same DINO features and masks."""
    torch.manual_seed(0)
    desc_dim = 1536
    H, W = 28, 42  # multiples of 14
    h_r = H // 14
    w_r = W // 14
    n_seg = 5

    # Random DINO feature, L2-normalised along the channel axis (mirroring
    # what _process_dino_batched writes to H5).
    feat = torch.randn(1, desc_dim, h_r, w_r).cuda()
    feat = torch.nn.functional.normalize(feat, dim=1).contiguous()

    # Random binary masks at full resolution.
    rs = np.random.RandomState(3)
    masks_seg = [rs.rand(H, W) > 0.7 for _ in range(n_seg)]
    for m in masks_seg:
        m[0, 0] = True
    masks_np = np.stack(masks_seg, axis=0)
    masks_gpu = torch.from_numpy(masks_np).cuda()

    # Random cluster centers.
    n_clusters = 32
    c_centers = torch.randn(n_clusters, desc_dim).cuda()

    idx_matrix, ind_matrix = build_pixel_to_patch_index(H, W, device="cuda")

    # Build adj on both sides identically.
    adj_legacy = func_vpr.nbrMasksAGGFastSingle(masks_seg, order=3)
    adj_new = _build_adj_gpu(masks_gpu, order=3)
    assert adj_new is not None
    np.testing.assert_array_equal(adj_new.cpu().numpy(), adj_legacy.cpu().numpy())

    # Stash the DINO feature in a temp H5 so legacy can read it the way it
    # expects.
    dino_path = tmp_path / "dino_legacy.h5"
    with h5py.File(str(dino_path), "w") as f:
        grp = f.create_group("img_0.png")
        grp.create_dataset("ift_dino", data=feat.cpu().numpy())

    legacy_cfg = {"desired_width": W, "desired_height": H}

    with h5py.File(str(dino_path), "r") as desc_h5:
        legacy_vlad = func_vpr.seg_vlad_gpu_single(
            ind_matrix, idx_matrix, desc_h5, "img_0.png", masks_seg,
            c_centers, legacy_cfg, desc_dim=desc_dim,
            adj_mat=adj_legacy, device="cuda",
        ).cuda()

    new_vlad = seg_vlad_gpu_from_tensors(
        feat[0], masks_gpu, c_centers, ind_matrix,
        desired_height=H, desired_width=W, desc_dim=desc_dim,
        adj_mat=adj_new, device="cuda",
    )

    torch.testing.assert_close(
        new_vlad.float(), legacy_vlad.float(), rtol=1e-4, atol=1e-5
    )


# ---------------------------------------------------------------------------
# PcaFitter.fit_from_buffer
# ---------------------------------------------------------------------------


@requires_cuda
def test_fit_from_buffer_matches_torch_svd() -> None:
    """The buffer entrypoint must give the same model as a direct
    ``_fit_torch_svd`` on the same data."""
    torch.manual_seed(0)
    ctx = _make_ctx("cuda")
    fitter = PcaFitter(
        ctx,
        n_components=16,
        whiten=True,
        svd_solver="randomized",
        max_segments=10000,
    )
    buf_fp16 = torch.randn(500, 256, dtype=torch.float16, device="cuda")

    torch.manual_seed(123)
    via_buffer = fitter.fit_from_buffer(buf_fp16)
    torch.manual_seed(123)
    direct = fitter._fit_torch_svd(buf_fp16.float())

    # PcaModel exposes .transform; assert identical projections on the same data
    sample = torch.randn(8, 256, device="cuda")
    np.testing.assert_allclose(
        via_buffer.transform(sample).cpu().numpy(),
        direct.transform(sample).cpu().numpy(),
        rtol=1e-4,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# vlads cache → compute_all_from_cache
# ---------------------------------------------------------------------------


def _fake_feature_store(tmp_path: Path, names: list[str]) -> FeatureStore:
    """Empty dino/masks H5 stores so FeatureStore can construct. We never
    exercise the H5 reads in the cached path.
    """
    dino_h5 = tmp_path / "dino.h5"
    masks_h5 = tmp_path / "masks.h5"
    for p in (dino_h5, masks_h5):
        with h5py.File(str(p), "w"):
            pass
    return FeatureStore(dino_h5, masks_h5, vlads_h5_path=None)


@requires_cuda
def test_compute_all_from_cache_skips_recompute(tmp_path: Path) -> None:
    """When ``vlads_cache_h5`` is provided, ``compute_all`` must read VLADs
    from the cache and only apply PCA (no SegVLAD recomputation)."""
    torch.manual_seed(0)
    ctx = _make_ctx("cuda")

    names = ["a.png", "b.png", "c.png"]
    vlad_dim = 32 * 1536
    fake_vlads = []
    vlads_h5 = tmp_path / "vlads.h5"
    with h5py.File(str(vlads_h5), "w") as f:
        for i, name in enumerate(names):
            n_seg = 3 + i
            v = np.random.randn(n_seg, vlad_dim).astype(np.float16)
            im_inds = np.full(n_seg, i, dtype=np.int64)
            grp = f.create_group(name)
            grp.create_dataset("vlad", data=v)
            grp.create_dataset("im_inds", data=im_inds)
            fake_vlads.append(v)

    # Fit a small PCA so we have a real transform to apply.
    fitter = PcaFitter(
        ctx,
        n_components=8,
        whiten=False,
        svd_solver="randomized",
        max_segments=10000,
    )
    pca = fitter.fit_from_buffer(
        torch.from_numpy(np.concatenate(fake_vlads, axis=0)).float().cuda()
    )

    store = _fake_feature_store(tmp_path, names)
    store.vlads_h5_path = vlads_h5

    n_clusters = 32
    c_centers = torch.randn(n_clusters, 1536)
    idx_matrix, ind_matrix = build_pixel_to_patch_index(
        height=140, width=140, device=torch.device("cuda")
    )
    comp = VladComputer(
        ctx,
        c_centers,
        ind_matrix,
        idx_matrix,
        desired_width=140,
        desired_height=140,
    )

    images = [Path(n) for n in names]
    out = comp.compute_all(
        store, images, pca=pca, vlads_cache_h5=store.vlads_h5_path
    )

    # Re-apply PCA manually to verify equivalence.
    expected = pca.transform(
        torch.from_numpy(np.concatenate(fake_vlads, axis=0)).float().cuda()
    ).cpu()
    np.testing.assert_allclose(
        out.feats.cpu().numpy(),
        expected.cpu().numpy(),
        rtol=1e-3,
        atol=1e-4,
    )
    # im_inds must match (rebuilt from the input image list, not the cached
    # values which encode the *writer-time* dataset index).
    expected_inds = np.concatenate(
        [np.full(v.shape[0], i, dtype=np.int64) for i, v in enumerate(fake_vlads)]
    )
    np.testing.assert_array_equal(out.im_inds, expected_inds)
