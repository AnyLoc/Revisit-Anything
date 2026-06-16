from revisit_anything.revisit_anything import RevisitAnything, create_pairs
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from revisit_anything.config import Config
from pathlib import Path

cs = ConfigStore.instance()
cs.store(name="default", node=Config)


@hydra.main(config_name="default")
def main(cfg: DictConfig):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    revisit = RevisitAnything(cfg, image_folder_name="images", output_folder_name="output")
    query_images = list((revisit._workdir / "images").glob("*"))
    current_vlads = revisit(query_images)
    if current_vlads is None:
        logger.error("Failed to compute VLADs for current images.")
        return
    _, txt_path = create_pairs(
        current_vlads,
        None,
        query_images, Path("/root/polish-pipeline/"),
        cfg.runtime, cfg.top_n, cfg.seg_topk, current_prefix="images", past_prefix="images_ref"
    )


if __name__ == "__main__":
    main()
