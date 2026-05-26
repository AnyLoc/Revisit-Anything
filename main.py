from revisit_anything.revisit_anything import RevisitAnything
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from revisit_anything.config import Config, RuntimeConfig

cs = ConfigStore.instance()
cs.store(name="default", node=Config)
@hydra.main(config_name="default")
def main(cfg: DictConfig):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    revisit = RevisitAnything(cfg, image_folder_name="images", output_folder_name="output")
    query_images = list((revisit._workdir / "images").glob("*"))
    revisit(query_images)

if __name__ == "__main__":
    main()
