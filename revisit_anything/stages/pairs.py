from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from loguru import logger

import revisit_anything.utils as utils
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
        past_vlads: AllVlads | None,
        query_images: list[Path],
        output_folder: Path,
        *,
        current_prefix: str = "",
        past_prefix: str = "",
        normalize: bool = True,
    ) -> tuple[Path, Path]:
        feats_np = vlads.feats.cpu().numpy()
        past_feats_np = past_vlads.feats.cpu().numpy() if past_vlads is not None else None
        if normalize:
            feats_np = utils.normalizeFeat(feats_np)
            if past_feats_np is not None:
                past_feats_np = utils.normalizeFeat(past_feats_np)

        if past_feats_np is not None:
            assert feats_np.shape[1] == past_feats_np.shape[1], (
                f"VLAD descriptor dim mismatch: current={feats_np.shape[1]} vs past={past_feats_np.shape[1]}"
            )

        index = self._make_index(feats_np.shape[1])
        index.add(feats_np)
        if past_feats_np is not None:
            index.add(past_feats_np)

        n_cur = len(vlads.images)
        if past_vlads is not None:
            all_images = vlads.images + past_vlads.images
            combined_im_inds = np.concatenate([
                vlads.im_inds,
                past_vlads.im_inds + n_cur,
            ])
        else:
            all_images = vlads.images
            combined_im_inds = vlads.im_inds

        # Map each query image to its index in the full current image list so we
        # can locate its segments in ``vlads.feats`` / ``vlads.im_inds``.
        cur_index_by_name = {img.name: i for i, img in enumerate(vlads.images)}
        query_idx_in_cur = [cur_index_by_name[q.name] for q in query_images]

        seg_range_full = vlads.seg_range(n_cur)
        seg_range_query = [seg_range_full[i] for i in query_idx_in_cur]
        query_seg_idx = np.concatenate(seg_range_query) if seg_range_query else np.empty(0, dtype=int)

        # ``get_matches`` indexes ``matches`` rows via segRangeQuery, so the
        # ranges must be expressed in the local 0..M-1 layout of ``q_feats``.
        seg_range_query_local: list[np.ndarray] = []
        cursor = 0
        for r in seg_range_query:
            seg_range_query_local.append(np.arange(cursor, cursor + len(r)))
            cursor += len(r)

        q_feats = feats_np[query_seg_idx] if len(query_seg_idx) else feats_np[:0]
        sims, matches = index.search(q_feats, self.seg_topk)
        logger.info(f"faiss search: sims={sims.shape}, matches={matches.shape}")

        # Same Borda aggregation as the original ``create_pairs``: take the
        # top-50 per-segment matches, weight by ``2 - L2`` similarity, then
        # vote at image level.
        sims_50 = 2 - sims[:, :50]
        matches_50 = matches[:, :50]
        dummy_gt = [[] for _ in range(len(query_images))]
        preds = utils.get_matches(
            matches_50,
            dummy_gt,
            sims_50,
            seg_range_query_local,
            combined_im_inds,
            n=self.top_n + 1,
            method="max_seg_topk_wt_borda_Im",
        )

        return self._write(
            preds,
            query_idx_in_cur,
            all_images,
            n_cur,
            current_prefix,
            past_prefix,
            output_folder,
        )

    def _write(
        self,
        preds,
        query_idx_in_cur: list[int],
        all_images: list[Path],
        n_cur: int,
        current_prefix: str,
        past_prefix: str,
        output_folder: Path,
    ) -> tuple[Path, Path]:
        def _label(j: int) -> str:
            img = all_images[j]
            prefix = current_prefix if j < n_cur else past_prefix
            return f"{prefix}{img.name}"

        pair_map: dict[str, list[str]] = {}
        pair_set: set[tuple[str, str]] = set()
        for i, self_idx in enumerate(query_idx_in_cur):
            nbrs = [int(j) for j in preds[i] if int(j) != self_idx][: self.top_n]
            q_label = _label(self_idx)
            pair_map[q_label] = [_label(j) for j in nbrs]
            for j in nbrs:
                a, b = sorted([q_label, _label(j)])
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
