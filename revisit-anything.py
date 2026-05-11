import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from hydra_config import Config

import func_vpr
from loguru import logger
from natsort import natsorted
from pathlib import Path



cs = ConfigStore.instance()
cs.store(name="default", node=Config)

@hydra.main(config_name="default")
def main(cfg: DictConfig):
    cfg = OmegaConf.structured(cfg)
    workdir = Path(cfg.dataset_base_dir)/cfg.dataset_name/cfg.scene_name
    output_folder = workdir/f"outputs"
    image_folder = workdir/"images"
    image_feature_dict_folder = output_folder/"image_feature_dicts"
    image_feature_dict_folder.mkdir(parents=True, exist_ok=True)

    # TODO start end step for ims
    ims_sidx, ims_eidx, ims_step = 0, None, 1
    ims = natsorted(list(image_folder.glob("*")))
    ims = ims[ims_sidx:ims_eidx][::ims_step]

    # fastsam extraction
    # dino extraction
    logger.info("Starting DINO extraction with the following configuration:")
    cfg_dino = { "desired_width": cfg.desired_width, "desired_height": cfg.desired_height, "detect": 'dino', "use_sam": True, "class_threshold": 0.9, \
                "desired_feature": 0, "query_type": 'text', "sort_by": 'area', "use_16bit": False, "use_cuda": True,\
                "dino_strides": 4, "use_traced_model": False, 
                "rmin":0, "DAStoreFull":False, "dinov2": True, "wrap":False, "resize": True} # robohop specifc params

    dino_model = func_vpr.loadDINO(cfg_dino, cfg.dino_model, device="cuda")
    h5FullPathDINO = image_feature_dict_folder / f"dino_{cfg.desired_width}.h5"
    func_vpr.process_dino_ft_to_h5(h5FullPathDINO,cfg_dino,ims,dino_model,dataDir=str(image_folder))
    logger.info("DINO extraction completed.")

    # sam extraction
    logger.info("Starting SAM extraction with the following configuration:")
    cfg_sam = { "desired_width": cfg.desired_width, "desired_height": cfg.desired_height, "detect": 'dino', "use_sam": True, "class_threshold": 0.9, \
                "desired_feature": 0, "query_type": 'text', "sort_by": 'area', "use_16bit": False, "use_cuda": True,\
                "dino_strides": 4, "use_traced_model": False, 
                "rmin":0, "DAStoreFull":False, "dinov2": True, "wrap":False, "resize": True} # robohop specifc params

    SAM = func_vpr.loadSAM(cfg.sam_checkpoint, cfg_sam, device="cuda")
    h5FullPathSAM = image_feature_dict_folder / f"sam_{cfg.desired_width}.h5"
    func_vpr.process_SAM_to_h5(h5FullPathSAM, cfg_sam,ims,SAM,dataDir=str(image_folder))
    logger.info("SAM extraction completed.")

if __name__ == "__main__":
    main()
