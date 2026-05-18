from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from loguru import logger

import func_vpr
from revisit_anything.runtime import RuntimeCtx
from revisit_anything.stages.vlad import AllVlads


class PairBuilder:
    """Stage 3 (pair construction). Consumes an in-memory ``AllVlads`` —
    no H5 reopen and no second VLAD pass."""

    def __init__(self, ctx: RuntimeCtx, *, top_n: int, seg_topk: int):
        self.ctx = ctx
        self.top_n = top_n
        self.seg_topk = seg_topk

    def _make_index(self, dim: int) -> faiss.Index:
        cpu_index = faiss.IndexFlatL2(dim)
        if self.ctx.faiss_backend == "cpu":
            return cpu_index
        if self.ctx.faiss_backend == "gpu":
            if not hasattr(faiss, "StandardGpuResources"):
                raise RuntimeError(
                    "faiss_backend='gpu' requested but faiss-gpu is not installed. "
                    "Install with: conda install -c pytorch faiss-gpu"
                )
            self._gpu_res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(self._gpu_res, 0, cpu_index)
        raise ValueError(f"Unknown faiss_backend: {self.ctx.faiss_backend!r}")

    def build(
        self,
        vlads: AllVlads,
        images: list[Path],
        output_folder: Path,
        *,
        normalize: bool = True,
    ) -> tuple[Path, Path]:
        feats_np = vlads.feats.cpu().numpy()
        if normalize:
            feats_np = func_vpr.normalizeFeat(feats_np)

        index = self._make_index(feats_np.shape[1])
        index.add(feats_np)
        sims, matches = index.search(feats_np, self.seg_topk)
        logger.info(f"faiss search: sims={sims.shape}, matches={matches.shape}")

        # Same Borda aggregation as the original ``create_pairs``: take the
        # top-50 per-segment matches, weight by ``2 - L2`` similarity, then
        # vote at image level.
        sims_50 = 2 - sims[:, :50]
        matches_50 = matches[:, :50]
        seg_range = vlads.seg_range(len(images))
        dummy_gt = [[] for _ in range(len(images))]
        preds = func_vpr.get_matches(
            matches_50, dummy_gt, sims_50, seg_range, vlads.im_inds,
            n=self.top_n + 1, method="max_seg_topk_wt_borda_Im",
        )

        return self._write(preds, images, output_folder)

    def _write(
        self, preds, images: list[Path], output_folder: Path
    ) -> tuple[Path, Path]:
        pair_map = {}
        pair_set = set()
        for i, img in enumerate(images):
            nbrs = [int(j) for j in preds[i] if int(j) != i][: self.top_n]
            pair_map[img.name] = [images[j].name for j in nbrs]
            for j in nbrs:
                a, b = sorted([img.name, images[j].name])
                pair_set.add((a, b))

        json_path = output_folder / f"pairs_top{self.top_n}.json"
        with open(json_path, "w") as f:
            json.dump(pair_map, f, indent=2)
        logger.info(f"Wrote {json_path}")

        txt_path = output_folder / f"pairs_top{self.top_n}.txt"
        with open(txt_path, "w") as f:
            for a, b in sorted(pair_set):
                f.write(f"{a} {b}\n")
        logger.info(f"Wrote {txt_path} ({len(pair_set)} unique pairs)")
        return json_path, txt_path
