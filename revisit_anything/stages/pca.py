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
    ):
        self.ctx = ctx
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.max_segments = max_segments

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
        """GPU-resident PCA via low-rank SVD. Compatible interface with sklearn
        (mean centering + optional whitening), but components are computed
        directly on ``ctx.device`` so no host round-trip is required.
        """
        x = x.to(self.ctx.device, dtype=torch.float32)
        n = x.shape[0]
        mean = x.mean(dim=0)
        x_centered = x - mean
        # Oversample for stability of randomized SVD.
        q = min(self.n_components + 10, min(x_centered.shape))
        _, S, V = torch.svd_lowrank(x_centered, q=q)
        components = V[:, : self.n_components].T.contiguous()  # [k, d]
        singular = S[: self.n_components]
        explained_variance = (singular**2) / max(n - 1, 1)
        total_variance = (x_centered**2).sum() / max(n - 1, 1)
        evr = (explained_variance / total_variance).cpu()
        whiten_scale = torch.sqrt(explained_variance).cpu() if self.whiten else None
        state = _TorchPcaState(
            mean=mean.cpu(),
            components=components.cpu(),
            whiten_scale=whiten_scale,
            explained_variance_ratio_=evr,
        )
        logger.info(f"Explained variance ratio: {evr}")
        return PcaModel(state, device=self.ctx.device, backend="torch_svd")

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
