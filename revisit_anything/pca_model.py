from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from sklearn import config_context

if TYPE_CHECKING:
    from revisit_anything.runtime import RuntimeCtx


@dataclass
class _TorchPcaState:
    """Serializable PCA state for the torch SVD backend.

    Tensors are stored on CPU so the pickle is portable; ``PcaModel``
    re-uploads to its bound device on load.
    """
    mean: torch.Tensor                  # [d_in]
    components: torch.Tensor            # [n_components, d_in]
    whiten_scale: torch.Tensor | None   # [n_components] or None
    explained_variance_ratio_: torch.Tensor   # [n_components]


class PcaModel:
    """Backend-agnostic wrapper around a fitted PCA model.

    ``backend='sklearn'`` wraps an ``sklearn.decomposition.PCA``.
    ``backend='torch_svd'`` wraps a ``_TorchPcaState`` and runs the projection
    entirely on the bound device (no GPU→CPU round-trip).
    """

    def __init__(self, fitted, *, device: torch.device, backend: str):
        self._fitted = fitted
        self.device = device
        self.backend = backend
        if backend == "torch_svd":
            assert isinstance(fitted, _TorchPcaState)
            self._mean = fitted.mean.to(device)
            self._components = fitted.components.to(device)
            self._whiten_scale = (
                fitted.whiten_scale.to(device)
                if fitted.whiten_scale is not None
                else None
            )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "sklearn":
            with config_context(array_api_dispatch=True):
                return self._fitted.transform(x.float()).to(self.device)
        if self.backend == "torch_svd":
            x = x.to(self.device, dtype=torch.float32)
            x = x - self._mean
            out = x @ self._components.T
            if self._whiten_scale is not None:
                out = out / self._whiten_scale
            return out
        raise NotImplementedError(f"PcaModel backend {self.backend!r}")

    def save(self, path: Path) -> None:
        # sklearn backend: save the raw sklearn PCA so the pickle stays
        # compatible with legacy ``apply_pca_transform_from_pkl`` callers.
        # torch_svd backend: pickle the dataclass — only readable through
        # ``PcaModel.load``.
        with open(path, "wb") as f:
            pickle.dump(self._fitted, f)

    @classmethod
    def load(cls, path: Path, ctx: "RuntimeCtx") -> "PcaModel":
        with open(path, "rb") as f:
            fitted = pickle.load(f)
        backend = "torch_svd" if isinstance(fitted, _TorchPcaState) else "sklearn"
        return cls(fitted, device=ctx.device, backend=backend)

    @property
    def explained_variance_ratio_(self):
        if self.backend == "torch_svd":
            return self._fitted.explained_variance_ratio_
        return self._fitted.explained_variance_ratio_

    @property
    def n_components(self) -> int:
        if self.backend == "torch_svd":
            return int(self._fitted.components.shape[0])
        return int(self._fitted.n_components_)
