from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch
from loguru import logger
from sklearn import config_context
from sklearn.decomposition import PCA

from revisit_anything.feature_store import FeatureStore
from revisit_anything.pca_model import PcaModel, _TorchPcaState
from revisit_anything.runtime import RuntimeCtx
from revisit_anything.stages.vlad import ImageVlad, VladComputer


class PcaFitter:
    """Stage 2 (PCA fit). Consumes the same ``ImageVlad`` stream that the
    pair-building stage will iterate over, so the orchestrator can either
    fit fresh or load an existing pickle without recomputing VLADs.
    """

    def __init__(
        self,
        ctx: RuntimeCtx,
        *,
        n_components: int,
        whiten: bool,
        svd_solver: str,
        max_segments: int,
        svd_chunk_size: int = 4096,
    ):
        self.ctx = ctx
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.max_segments = max_segments
        self.svd_chunk_size = svd_chunk_size

    def fit_streaming(
        self,
        vlad_iter: Iterator[ImageVlad],
        total_segments: int,
    ) -> PcaModel:
        sampling_ratio = min(1.0, self.max_segments / max(total_segments, 1))
        accumulated = 0
        sampled: list[torch.Tensor] = []

        for item in vlad_iter:
            gd = item.vlad
            n = gd.shape[0]
            k = int(n * sampling_ratio)
            if k > 0:
                idx = torch.randperm(n)[:k]
                sampled.append(gd[idx])
                accumulated += k
            if accumulated >= self.max_segments:
                break

        if not sampled:
            raise RuntimeError("No segments accumulated for PCA fit")
        seg_ft_vlad = torch.cat(sampled, dim=0)
        logger.info(f"Fitting PCA on {accumulated} segments (sampled from {total_segments} total)")
        logger.info(f"PCA fit: {seg_ft_vlad.shape}")

        if self.ctx.pca_backend == "sklearn":
            with config_context(array_api_dispatch=True):
                pca = PCA(
                    n_components=self.n_components,
                    whiten=self.whiten,
                    svd_solver=self.svd_solver,
                )
                pca.fit(seg_ft_vlad)
            logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")
            return PcaModel(pca, device=self.ctx.device, backend="sklearn")

        if self.ctx.pca_backend == "torch_svd":
            return self._fit_torch_svd(seg_ft_vlad)

        raise NotImplementedError(f"pca_backend={self.ctx.pca_backend!r}")

    def fit_from_buffer(self, buffer: torch.Tensor) -> PcaModel:
        """Fit PCA from a pre-sampled tensor (e.g. a GPU reservoir).

        Used by :class:`FusedFeaturePcaStage` where the per-image VLADs are
        reservoir-sampled in-flight, avoiding the second pass that
        :meth:`fit_streaming` does when consuming :class:`VladComputer.iter`.
        ``buffer`` may be fp16 (typical for a GPU reservoir); we promote to
        fp32 because :func:`torch.svd_lowrank` requires float.
        """
        if buffer.numel() == 0:
            raise RuntimeError("Empty buffer for PCA fit")
        logger.info(f"PCA fit from buffer: {tuple(buffer.shape)} dtype={buffer.dtype}")
        if self.ctx.pca_backend == "torch_svd":
            return self._fit_torch_svd(buffer.float())
        if self.ctx.pca_backend == "sklearn":
            with config_context(array_api_dispatch=True):
                pca = PCA(
                    n_components=self.n_components,
                    whiten=self.whiten,
                    svd_solver=self.svd_solver,
                )
                pca.fit(buffer.float().cpu())
            logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")
            return PcaModel(pca, device=self.ctx.device, backend="sklearn")
        raise NotImplementedError(f"pca_backend={self.ctx.pca_backend!r}")

    def _fit_torch_svd(self, x: torch.Tensor) -> PcaModel:
        """GPU-resident PCA via chunked randomized SVD.

        Keeps ``x`` in its original dtype (typically fp16 from the reservoir)
        and only upcasts one row-chunk to fp32 at a time, so the (n, d)
        matrix is never fully materialised in fp32. Mean centering and the
        two randomized-SVD matmuls (Y = (X-μ)Ω, B = Qᵀ(X-μ)) are streamed
        over row chunks of ``svd_chunk_size``. Numerically equivalent to
        the previous ``torch.svd_lowrank`` call up to the random sketch.
        """
        device = self.ctx.device
        x = x.to(device)  # keep dtype (fp16 reservoir stays fp16 on GPU)
        n, d = x.shape
        chunk = max(1, self.svd_chunk_size)
        q = min(self.n_components + 10, n, d)  # oversample for stability

        # 1) mean (fp32, d)
        mean = torch.zeros(d, device=device, dtype=torch.float32)
        for i in range(0, n, chunk):
            mean += x[i : i + chunk].float().sum(dim=0)
        mean /= n

        # 2) total_variance via streamed (x - mean) squared sum; avoids the
        #    (n, d) fp32 ``x_centered**2`` temporary.
        total_var = torch.zeros((), device=device, dtype=torch.float32)
        for i in range(0, n, chunk):
            c = x[i : i + chunk].float() - mean
            total_var += c.pow(2).sum()
        total_var = total_var / max(n - 1, 1)

        # 3) Stage 1: Y = (x - mean) @ Ω, accumulated chunk-by-chunk.
        Omega = torch.randn(d, q, device=device, dtype=torch.float32)
        Y = torch.empty(n, q, device=device, dtype=torch.float32)
        for i in range(0, n, chunk):
            c = x[i : i + chunk].float() - mean
            Y[i : i + chunk] = c @ Omega
        del Omega

        # 4) Range basis Q via QR.
        Q, _ = torch.linalg.qr(Y)
        del Y

        # 5) Stage 2: B = Qᵀ (x - mean), accumulated chunk-by-chunk.
        B = torch.zeros(q, d, device=device, dtype=torch.float32)
        for i in range(0, n, chunk):
            c = x[i : i + chunk].float() - mean
            B += Q[i : i + chunk].T @ c
        del Q

        # 6) Small (q, d) SVD recovers the top components.
        _U, S, Vh = torch.linalg.svd(B, full_matrices=False)
        del B
        components = Vh[: self.n_components].contiguous()  # [k, d]
        singular = S[: self.n_components]
        explained_variance = (singular**2) / max(n - 1, 1)
        evr = (explained_variance / total_var).cpu()
        whiten_scale = torch.sqrt(explained_variance).cpu() if self.whiten else None
        state = _TorchPcaState(
            mean=mean.cpu(),
            components=components.cpu(),
            whiten_scale=whiten_scale,
            explained_variance_ratio_=evr,
        )
        logger.info(f"Explained variance ratio: {evr}")
        return PcaModel(state, device=device, backend="torch_svd")

    def fit_or_load(
        self,
        store: FeatureStore,
        images: list,
        vlad_comp: VladComputer,
        cache_path: Path,
        *,
        timer=None,
    ) -> PcaModel:
        """Cached entrypoint: load from ``cache_path`` if it exists, otherwise
        fit on a streamed sample of per-image VLADs and persist.
        """
        if cache_path.is_file():
            logger.info(f"PCA model cache hit: {cache_path}")
            return PcaModel.load(cache_path, self.ctx)

        def _timed(name):
            import contextlib

            return timer.timeit_block(name) if timer is not None else contextlib.nullcontext()

        with _timed("Counting total number of segments in dataset"):
            total = store.count_total_segments([p.name for p in images])
        with _timed("Fitting PCA model"):
            model = self.fit_streaming(vlad_comp.iter(store, images, desc="Processing images..."), total)
        with _timed("Saving PCA model to file"):
            model.save(cache_path)
        logger.info(f"PCA model saved to {cache_path}")
        return model
