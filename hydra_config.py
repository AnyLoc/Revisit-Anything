from typing import Literal, Optional
from dataclasses import dataclass, field

@dataclass(frozen=True)
class PCAConfig:
    max_segments: int = 50000
    lower_dim: int = 1024
    whiten: bool = True
    # svd_solver: Literal["auto", "full", "arpack", "randomized"] = "arpack"
    svd_solver: str = "arpack"
    order: Optional[int] = 3
    min_area: int = 0

@dataclass(frozen=True)
class VLADConfig:
    cache_dir: str = "/workspace/Revisit-Anything/cache"
    number_clusters: int = 32
    desc_layer: int = 31
    # desc_facet: Literal["query", "key", "value", "token"] = "value"
    desc_facet: str = "value"
    desc_dim: int = 1536

    domain: str = "indoor"

@dataclass
class Config:
    dataset_base_dir: str = "/home/ubuntu/"
    dataset_name: str = "search-data"
    scene_name: str = "0"

    desired_width: int = 640
    desired_height: int = 320

    sam_checkpoint: str = "/home/ubuntu/search-data/models/segment-anything/sam_vit_h_4b8939.pth"
    fastsam_checkpoint: str = "/home/ubuntu/search-data/models/segment-anything/FastSAM-x.pt"
    dino_model: str = "dinov2_vitg14"

    mask_segmentation: str = "fastsam"
    # global_method_name: Literal["SegLoc", "VLAD", "PCA"] = "SegLoc"
    global_method_name: str = "SegLoc"
    pca_fit: bool = True

    vlad_config: VLADConfig = VLADConfig()
    pca_config: PCAConfig = PCAConfig()

    top_n: int = 10
    seg_topk: int = 200
