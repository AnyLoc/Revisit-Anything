import sys
from pathlib import Path

import hydra
import torch
from hydra.core.config_store import ConfigStore
from loguru import logger
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf

from hydra_config import Config
from utilities import VLAD

from revisit_anything.runtime import resolve_runtime
from revisit_anything.timer import Timer
from revisit_anything.vlad_index import build_pixel_to_patch_index, load_cluster_centers
from revisit_anything.stages.extract import FeatureExtractor
from revisit_anything.stages.vlad import VladComputer
from revisit_anything.stages.pca import PcaFitter
from revisit_anything.stages.pairs import PairBuilder


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


def _list_images(image_folder: Path, top_n: int) -> list[Path]:
    images = natsorted(list(image_folder.glob("*")))
    n = len(images)
    if n <= 2:
        raise ValueError(
            f"Expected >2 images in {image_folder}, found {n}."
        )
    if top_n >= n:
        raise ValueError(
            f"top_n={top_n} must be < number of images ({n})."
        )
    return images


@hydra.main(config_name="default")
def main(cfg: DictConfig):
    cfg = OmegaConf.structured(cfg)
    workdir = Path(cfg.dataset_base_dir) / cfg.dataset_name / cfg.scene_name
    image_folder = workdir / "images"
    output_folder = workdir / "outputs"
    output_folder.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(cfg.dataset_base_dir)

    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level: <8} | [scene={extra[scene]}] {message}")
    logger.configure(extra={"scene": cfg.scene_name})

    images = _list_images(image_folder, cfg.top_n)
    timer = Timer(file_name=output_folder / "timing_info.json")
    ctx = resolve_runtime(cfg.runtime)

    extractor = FeatureExtractor(
        ctx,
        dino_model=cfg.dino_model,
        segmenter=cfg.mask_segmentation,
        sam_checkpoint=cfg.sam_checkpoint,
        fastsam_checkpoint=cfg.fastsam_checkpoint,
        desired_width=cfg.desired_width,
        desired_height=cfg.desired_height,
        timer=timer,
    )
    try:
        with extractor.extract(image_folder, images, output_folder) as store:
            c_centers, c_centers_file = load_cluster_centers(cfg.vlad_config, device=ctx.device)
            idx_matrix, ind_matrix = build_pixel_to_patch_index(
                cfg.desired_height, cfg.desired_width, device=ctx.device,
            )
            with timer.timeit_block("VLAD initialization"):
                VLAD(cfg.vlad_config.number_clusters, desc_dim=None,
                     cache_dir=c_centers_file.parent).fit(None)

            vlad_comp = VladComputer(
                ctx, c_centers, ind_matrix, idx_matrix,
                desired_width=cfg.desired_width,
                desired_height=cfg.desired_height,
                desc_dim=cfg.vlad_config.desc_dim,
                order=cfg.pca_config.order if cfg.pca_fit else 0,
                min_area=cfg.pca_config.min_area,
            )

            if cfg.global_method_name != "SegLoc":
                raise NotImplementedError(
                    f"global_method_name={cfg.global_method_name!r}; only 'SegLoc' is supported"
                )

            pca_model = PcaFitter(
                ctx,
                n_components=cfg.pca_config.lower_dim,
                whiten=cfg.pca_config.whiten,
                svd_solver=cfg.pca_config.svd_solver,
                max_segments=cfg.pca_config.max_segments,
            ).fit_or_load(
                store, images, vlad_comp,
                cache_path=output_folder / f"pca_model_{cfg.vlad_config.domain}.pkl",
                timer=timer,
            )

            with timer.timeit_block("Computing VLADs for pair construction"):
                all_vlads = vlad_comp.compute_all(store, images, pca=pca_model)

            with timer.timeit_block("Creating pairs"):
                PairBuilder(ctx, top_n=cfg.top_n, seg_topk=cfg.seg_topk).build(
                    all_vlads, images, output_folder,
                )
    finally:
        timer.save_times()
        extractor.close()


if __name__ == "__main__":
    main()
