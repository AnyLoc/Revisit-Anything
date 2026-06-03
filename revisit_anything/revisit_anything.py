from pathlib import Path

import torch
from hydra.core.config_store import ConfigStore
from natsort import natsorted

from revisit_anything.config import Config, RuntimeConfig
from revisit_anything.runtime import resolve_runtime
from revisit_anything.timer import Timer
from revisit_anything.vlad_index import build_pixel_to_patch_index, load_cluster_centers
from revisit_anything.stages.extract import FeatureExtractor
from revisit_anything.stages.fused import FusedFeaturePcaStage
from revisit_anything.stages.vlad import AllVlads, VladComputer, VLAD
from revisit_anything.stages.pca import PcaFitter
from revisit_anything.stages.pairs import PairBuilder


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


def _list_images(image_folder: Path, top_n: int) -> list[Path]:
    images = natsorted(list(image_folder.glob("*")))
    n = len(images)
    if n <= 2:
        raise ValueError(f"Expected >2 images in {image_folder}, found {n}.")
    if top_n >= n:
        raise ValueError(f"top_n={top_n} must be < number of images ({n}).")
    return images


class RevisitAnything:
    def __init__(self, cfg: Config, image_folder_name: str, output_folder_name: str):
        self._cfg = cfg
        self._workdir = Path(self._cfg.dataset_base_dir) / self._cfg.dataset_name / self._cfg.scene_name
        print(self._workdir)
        self._image_folder = self._workdir / image_folder_name
        self._output_folder = self._workdir / output_folder_name
        self.pca_model_path = self._output_folder / f"pca_model_{self._cfg.vlad_config.domain}.pkl"
        self.feature_file = Path()
        self.mask_file = Path()
        self.timer = Timer(file_name=self._output_folder / "timing_info.json")

    def __call__(self, query_images: list[Path]) -> AllVlads | None:
        if not self._workdir.exists():
            return None

        self._output_folder.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(self._cfg.dataset_base_dir)

        images = _list_images(self._image_folder, self._cfg.top_n)
        ctx = resolve_runtime(self._cfg.runtime)

        if self._cfg.global_method_name != "SegLoc":
            raise NotImplementedError(
                f"global_method_name={self._cfg.global_method_name!r}; only 'SegLoc' is supported"
            )

        c_centers, c_centers_file = load_cluster_centers(self._cfg.vlad_config, device=ctx.device)
        idx_matrix, ind_matrix = build_pixel_to_patch_index(
            self._cfg.desired_height,
            self._cfg.desired_width,
            device=ctx.device,
        )
        with self.timer.timeit_block("VLAD initialization"):
            VLAD(self._cfg.vlad_config.number_clusters, desc_dim=None, cache_dir=c_centers_file.parent).fit(None)

        if self._cfg.fused_pipeline:
            all_vlads = self._run_fused(
                ctx,
                images,
                query_images,
                c_centers,
                ind_matrix,
                idx_matrix,
            )
        else:
            all_vlads = self._run_legacy(
                ctx,
                images,
                query_images,
                c_centers,
                ind_matrix,
                idx_matrix,
            )
        return all_vlads

    def _run_fused(
        self,
        ctx,
        images,
        query_images,
        c_centers,
        ind_matrix,
        idx_matrix,
    ) -> AllVlads:
        stage = FusedFeaturePcaStage(
            ctx,
            config=self._cfg,
            c_centers=c_centers,
            ind_matrix=ind_matrix,
            timer=self.timer,
        )
        try:
            store, pca_model = stage.run(
                self._image_folder,
                images,
                self._output_folder,
                self.pca_model_path,
            )
            with store:
                self.feature_file = store.dino_h5_path
                self.mask_file = store.masks_h5_path
                vlad_comp = VladComputer(
                    ctx,
                    c_centers,
                    ind_matrix,
                    idx_matrix,
                    desired_width=self._cfg.desired_width,
                    desired_height=self._cfg.desired_height,
                    desc_dim=self._cfg.vlad_config.desc_dim,
                    order=self._cfg.pca_config.order if self._cfg.pca_fit else 0,
                    min_area=self._cfg.pca_config.min_area,
                )
                with self.timer.timeit_block("Computing VLADs for pair construction"):
                    all_vlads = vlad_comp.compute_all(
                        store,
                        query_images,
                        pca=pca_model,
                        vlads_cache_h5=store.vlads_h5_path,
                    )
        finally:
            stage.close()
        return all_vlads

    def _run_legacy(
        self,
        ctx,
        images,
        query_images,
        c_centers,
        ind_matrix,
        idx_matrix,
    ) -> AllVlads:
        extractor = FeatureExtractor(
            ctx,
            dino_model=self._cfg.dino_model_name,
            segmenter=self._cfg.mask_segmentation,
            dino_checkpoint=self._cfg.dino_checkpoint,
            sam_checkpoint=self._cfg.sam_checkpoint,
            fastsam_checkpoint=self._cfg.fastsam_checkpoint,
            desired_width=self._cfg.desired_width,
            desired_height=self._cfg.desired_height,
            timer=self.timer,
            dino_batch_size=64,
        )
        try:
            with extractor.extract(self._image_folder, images, self._output_folder) as store:
                self.feature_file = store.dino_h5_path
                self.mask_file = store.masks_h5_path
                vlad_comp = VladComputer(
                    ctx,
                    c_centers,
                    ind_matrix,
                    idx_matrix,
                    desired_width=self._cfg.desired_width,
                    desired_height=self._cfg.desired_height,
                    desc_dim=self._cfg.vlad_config.desc_dim,
                    order=self._cfg.pca_config.order if self._cfg.pca_fit else 0,
                    min_area=self._cfg.pca_config.min_area,
                )
                pca_model = PcaFitter(
                    ctx,
                    n_components=self._cfg.pca_config.lower_dim,
                    whiten=self._cfg.pca_config.whiten,
                    svd_solver=self._cfg.pca_config.svd_solver,
                    max_segments=self._cfg.pca_config.max_segments,
                ).fit_or_load(
                    store,
                    images,
                    vlad_comp,
                    cache_path=self.pca_model_path,
                    timer=self.timer,
                )
                with self.timer.timeit_block("Computing VLADs for pair construction"):
                    all_vlads = vlad_comp.compute_all(store, query_images, pca=pca_model)
        finally:
            extractor.close()
        return all_vlads


def create_pairs(
    current_vlads: AllVlads,
    past_vlads: AllVlads | None,
    query_images: list[Path],
    output_folder: Path,
    runtime_cfg: RuntimeConfig,
    top_n: int,
    seg_topk: int,
    *,
    current_prefix: str = "",
    past_prefix: str = "",
) -> tuple[Path, Path]:
    pair_builder = PairBuilder(runtime_cfg, top_n=top_n, seg_topk=seg_topk)
    json_path, txt_path = pair_builder.build(
        current_vlads,
        past_vlads,
        query_images,
        output_folder,
        current_prefix=current_prefix,
        past_prefix=past_prefix,
    )
    return json_path, txt_path


# @hydra.main(config_name="default")
# def main(cfg: DictConfig):
# cfg = OmegaConf.structured(cfg)
# logger.remove()
# logger.add(sys.stderr, format="{time:HH:mm:ss} | {level: <8} | [scene={extra[scene]}] {message}")
# logger.configure(extra={"scene": cfg.scene_name})
# # RevisitAnything(cfg, "images", "outputs")


# if __name__ == "__main__":
# main()
