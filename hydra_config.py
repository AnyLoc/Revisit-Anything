from typing import Literal, Optional
from dataclasses import dataclass, field

@dataclass(frozen=True)
class PCAConfig:
    max_segments: int = 50000
    lower_dim: int = 1024
    whiten: bool = True
    svd_solver: Literal["auto", "full", "arpack", "randomized"] = "arpack"
    order: Optional[int] = 3

@dataclass(frozen=True)
class VLADConfig:
    cache_dir: str = "cache"
    number_clusters: int = 32
    desc_layer: int = 31
    desc_facet: Literal["query", "key", "value", "token"] = "value"
    domain: str = "indoor"

@dataclass
class Config:
    dataset_base_dir: str = "/home/ubuntu/"
    dataset_name: str = "search-data"
    scene_name: str = "0"

    desired_width: int = 640
    desired_height: int = 320
    sam_checkpoint: str = "/home/ubuntu/search-data/models/segment-anything/sam_vit_h_4b8939.pth"
    dino_model: str = "dinov2_vitg14"

    global_method_name: Literal["SegLoc", "VLAD", "PCA"] = "SegLoc"

    vlad_config: VLADConfig = VLADConfig()
    pca_config: PCAConfig = PCAConfig()

