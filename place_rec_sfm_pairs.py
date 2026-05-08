"""Generate top-N image pairs for SFM input via SegVLAD similarity.

For each image in a folder, finds the top-N most similar OTHER images
(self-match excluded) and writes:
  - pairs_topN.json : {image_filename: [top-N filenames]} (per-image map)
  - pairs_topN.txt  : COLMAP/hloc-style pairs file, one "imgA imgB" per line,
                      deduped (each unordered pair appears once)

Pre-requisites
--------------
1. The dataset must exist in place_rec_global_config.py with at least
   `data_subpath1_r`, `dino_h5_filename_r`, `masks_h5_filename_r`,
   `domain_vlad_cluster`, `cfg` set. For SFM the typical setup is to put
   ALL images into the ref folder; the query side is unused here.
2. SAM masks and DINO features must be extracted:
       python place_rec_SAM_DINO.py --dataset <name> --method SAM
       python place_rec_SAM_DINO.py --dataset <name> --method DINO
3. (Optional) If using a PCA experiment (e.g. exp0_global_SegLoc_VLAD_PCA_o3),
   the PCA model must be fit:
       python place_rec_pca.py --dataset <name> --experiment <exp> --vocab-vlad domain

Usage
-----
    python place_rec_sfm_pairs.py \
        --dataset my_data \
        --experiment exp0_global_SegLoc_VLAD_PCA_o3 \
        --vocab-vlad domain \
        --topn 20
"""
import json
import os
from enum import Enum
from os.path import join
from pathlib import Path
from typing import Optional

import faiss
import h5py
import numpy as np
import torch
import typer
from natsort import natsorted
from tqdm import tqdm
from typing_extensions import Annotated

import func_vpr
from place_rec_global_config import datasets, experiments, workdir_data


class VocabVlad(str, Enum):
    domain = "domain"
    map = "map"


def compute_seg_vlad(images, dino_h5_path, masks_h5_path, c_centers, cfg,
                     ind_matrix, idx_matrix, order, pca_model_path,
                     min_area=0, batch_size=100):
    """Run SegVLAD on each image. Returns (descriptors[N_seg, D], im_inds[N_seg])."""
    feats_list, pca_list, batch_buf = [], [], []
    im_inds = np.array([], dtype=int)

    dino_h5 = h5py.File(dino_h5_path, "r")
    masks_h5 = h5py.File(masks_h5_path, "r")
    try:
        for i, img_name in enumerate(tqdm(images, desc="SegVLAD")):
            masks_seg = func_vpr.preload_masks(masks_h5, img_name)
            inds_i, _, seg_mask_i = func_vpr.getIdxSingleFast(i, masks_seg, minArea=min_area)
            im_inds = np.concatenate((im_inds, inds_i))
            adj = func_vpr.nbrMasksAGGFastSingle(masks_seg, order) if order else None
            gd = func_vpr.seg_vlad_gpu_single(
                ind_matrix, idx_matrix, dino_h5, img_name, seg_mask_i,
                c_centers, cfg, desc_dim=1536, adj_mat=adj,
            )
            if pca_model_path is not None:
                batch_buf.append(gd)
                if (i + 1) % batch_size == 0 or (i + 1) == len(images):
                    batch_t = torch.cat(batch_buf, dim=0)
                    batch_buf = []
                    pca_list.append(func_vpr.apply_pca_transform_from_pkl(batch_t, pca_model_path))
            else:
                feats_list.append(gd)
    finally:
        dino_h5.close()
        masks_h5.close()

    if pca_model_path is not None:
        return torch.cat(pca_list, dim=0), im_inds
    return torch.cat(feats_list, dim=0), im_inds


def main(
    dataset: Annotated[str, typer.Option(
        "--dataset", "-d",
        help="Dataset key in place_rec_global_config.py.")],
    experiment: Annotated[str, typer.Option(
        "--experiment", "-e",
        help="Experiment key (must be a SegLoc-type entry).")] = "exp0_global_SegLoc_VLAD_PCA_o3",
    vocab_vlad: Annotated[VocabVlad, typer.Option(
        "--vocab-vlad",
        help="Which VLAD vocabulary to use.")] = VocabVlad.domain,
    topn: Annotated[int, typer.Option(
        "--topn", "-n",
        min=1,
        help="Top-N matches per image (excluding self).")] = 20,
    output_dir: Annotated[Optional[Path], typer.Option(
        "--output-dir", "-o",
        help="Output directory. Default: {workdir}/results/sfm_pairs/")] = None,
    seg_topk: Annotated[int, typer.Option(
        "--seg-topk",
        min=1,
        help="Per-segment faiss top-k before image-level aggregation.")] = 200,
) -> None:
    """Generate top-N image-pair list for SFM via SegVLAD similarity."""
    ds_cfg = datasets.get(dataset)
    if ds_cfg is None:
        raise typer.BadParameter(
            f"Dataset '{dataset}' not in place_rec_global_config.py", param_hint="--dataset"
        )
    ex_cfg = experiments.get(experiment)
    if ex_cfg is None:
        raise typer.BadParameter(
            f"Experiment '{experiment}' not in place_rec_global_config.py",
            param_hint="--experiment",
        )
    if ex_cfg.get("global_method_name") != "SegLoc":
        raise typer.BadParameter(
            "This script only supports SegLoc experiments.", param_hint="--experiment"
        )

    cfg = ds_cfg["cfg"]
    workdir = f"{workdir_data}/{dataset}/out"
    out_dir = str(output_dir) if output_dir is not None else join(workdir, "results", "sfm_pairs")
    os.makedirs(out_dir, exist_ok=True)

    domain = ds_cfg["domain_vlad_cluster"] if vocab_vlad == VocabVlad.domain else ds_cfg["map_vlad_cluster"]
    typer.echo(f"Using {vocab_vlad.value} vocabulary -> '{domain}'")

    c_centers_file = f"./cache/vocabulary/dinov2_vitg14/l31_value_c32/{domain}/c_centers.pt"
    if not os.path.isfile(c_centers_file):
        typer.echo(f"Cluster centers not found: {c_centers_file}", err=True)
        raise typer.Exit(code=1)
    c_centers = torch.load(c_centers_file)

    # Patch index matrix (matches place_rec_main.py)
    dh = cfg["desired_height"] // 14
    dw = cfg["desired_width"] // 14
    idx_matrix = np.empty((cfg["desired_height"], cfg["desired_width"], 2), dtype="int32")
    for i in range(cfg["desired_height"]):
        for j in range(cfg["desired_width"]):
            idx_matrix[i, j] = np.array([np.clip(i // 14, 0, dh - 1), np.clip(j // 14, 0, dw - 1)])
    ind_matrix = np.ravel_multi_index(idx_matrix.reshape(-1, 2).T, (dh, dw))
    ind_matrix = torch.tensor(ind_matrix, device="cuda")

    # SFM uses one set of images: read from the ref side.
    data_path = f"{workdir_data}/{dataset}/{ds_cfg['data_subpath1_r']}/"
    dino_h5_path = join(workdir, ds_cfg["dino_h5_filename_r"])
    masks_h5_path = join(workdir, ds_cfg["masks_h5_filename_r"])
    for p in (data_path, dino_h5_path, masks_h5_path):
        if not os.path.exists(p):
            typer.echo(f"Missing input: {p}\nRun place_rec_SAM_DINO.py first.", err=True)
            raise typer.Exit(code=1)

    images = natsorted(os.listdir(data_path))
    n_imgs = len(images)
    if n_imgs < 2:
        typer.echo(f"Need at least 2 images, got {n_imgs} in {data_path}", err=True)
        raise typer.Exit(code=1)
    if topn >= n_imgs:
        typer.echo(f"WARNING: topn={topn} >= n_images={n_imgs}; clamping to {n_imgs - 1}")
        topn = n_imgs - 1
    typer.echo(f"{n_imgs} images in {data_path}")

    # PCA model (only if the chosen experiment uses PCA)
    pca_model_path = None
    if ex_cfg.get("pca"):
        key = "pca_model_pkl" if vocab_vlad == VocabVlad.domain else "pca_model_pkl_map"
        pca_model_path = join(workdir, f"{dataset}{ex_cfg[key]}")
        if not os.path.isfile(pca_model_path):
            typer.echo(
                f"PCA model not found: {pca_model_path}\n"
                f"Run place_rec_pca.py --dataset {dataset} --experiment {experiment} "
                f"--vocab-vlad {vocab_vlad.value} first.",
                err=True,
            )
            raise typer.Exit(code=1)

    # 1) Compute SegVLAD for every image
    feats, im_inds = compute_seg_vlad(
        images, dino_h5_path, masks_h5_path, c_centers, cfg,
        ind_matrix, idx_matrix,
        order=ex_cfg.get("order", 0),
        pca_model_path=pca_model_path,
        min_area=ex_cfg.get("minArea", 0),
    )
    typer.echo(f"Total segments: {feats.shape[0]}, dim: {feats.shape[1]}")

    # 2) Build faiss index over all segments and search all-vs-all
    feats_np = feats.numpy()
    if pca_model_path is not None:
        feats_np = func_vpr.normalizeFeat(feats_np)
    index = faiss.IndexFlatL2(feats_np.shape[1])
    index.add(feats_np)
    sims, matches = index.search(feats_np, seg_topk)
    typer.echo(f"faiss search: sims={sims.shape}, matches={matches.shape}")

    # 3) Aggregate per-image top-(N+1) using the existing borda method,
    #    then strip self.
    sims_50 = 2 - sims[:, :50]
    matches_50 = matches[:, :50]
    seg_range = [np.where(im_inds == i)[0] for i in range(n_imgs)]
    dummy_gt = [[] for _ in range(n_imgs)]
    preds = func_vpr.get_matches(
        matches_50, dummy_gt, sims_50, seg_range, im_inds,
        n=topn + 1, method="max_seg_topk_wt_borda_Im",
    )

    pair_map = {}
    pair_set = set()
    for i, img in enumerate(images):
        nbrs = [int(j) for j in preds[i] if int(j) != i][:topn]
        pair_map[img] = [images[j] for j in nbrs]
        for j in nbrs:
            a, b = sorted([img, images[j]])
            pair_set.add((a, b))

    # 4) Write outputs
    json_path = join(out_dir, f"pairs_top{topn}.json")
    with open(json_path, "w") as f:
        json.dump(pair_map, f, indent=2)
    typer.echo(f"Wrote {json_path}")

    txt_path = join(out_dir, f"pairs_top{topn}.txt")
    with open(txt_path, "w") as f:
        for a, b in sorted(pair_set):
            f.write(f"{a} {b}\n")
    typer.echo(f"Wrote {txt_path}  ({len(pair_set)} unique pairs)")


if __name__ == "__main__":
    typer.run(main)
