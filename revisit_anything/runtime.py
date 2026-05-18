import contextlib
from dataclasses import dataclass
from typing import ContextManager

import torch
from loguru import logger


_DTYPE_BY_PRECISION = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class RuntimeCtx:
    device: torch.device
    dtype: torch.dtype
    faiss_backend: str
    pca_backend: str

    def autocast(self) -> ContextManager:
        if self.dtype == torch.float32:
            return contextlib.nullcontext()
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return torch.autocast(device_type="cpu", dtype=self.dtype)


def resolve_runtime(cfg) -> RuntimeCtx:
    """Resolve a Hydra RuntimeConfig into a concrete RuntimeCtx.

    ``cfg.device='auto'`` picks cuda if available else cpu.
    """
    raw = str(cfg.device)
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"runtime.device='{cfg.device}' requested but CUDA is unavailable"
        )

    precision = str(cfg.precision).lower()
    if precision not in _DTYPE_BY_PRECISION:
        raise ValueError(
            f"runtime.precision must be one of {list(_DTYPE_BY_PRECISION)}, "
            f"got {cfg.precision!r}"
        )

    ctx = RuntimeCtx(
        device=device,
        dtype=_DTYPE_BY_PRECISION[precision],
        faiss_backend=str(cfg.faiss_backend).lower(),
        pca_backend=str(cfg.pca_backend).lower(),
    )
    logger.info(
        f"runtime: device={ctx.device} dtype={ctx.dtype} "
        f"faiss={ctx.faiss_backend} pca={ctx.pca_backend}"
    )
    return ctx
