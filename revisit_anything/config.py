from typing import Literal, Optional
from dataclasses import dataclass, field


@dataclass
class RuntimeConfig:
    device: str = "auto"              # auto | cpu | cuda | cuda:N
    precision: str = "fp32"           # fp32 | fp16
    faiss_backend: str = "cpu"        # cpu | gpu
    pca_backend: str = "sklearn"      # sklearn | torch_svd

@dataclass
class PCAConfig:
    max_segments: int = 50000
    lower_dim: int = 1024
    whiten: bool = True
    # svd_solver: Literal["auto", "full", "arpack", "randomized"] = "arpack"
    svd_solver: str = "randomized"
    order: Optional[int] = 3
    min_area: int = 0

@dataclass
class VLADConfig:
    cache_dir: str = "Revisit-Anything/cache"
    number_clusters: int = 32
    desc_layer: int = 31
    # desc_facet: Literal["query", "key", "value", "token"] = "value"
    desc_facet: str = "value"
    desc_dim: int = 1536

    domain: str = "indoor"

@dataclass
class Config:
    dataset_base_dir: str = "/tmp/datasets/"
    dataset_name: str = "current"
    scene_name: str = "0"

    desired_width: int = 640
    desired_height: int = 320

    sam_checkpoint: str = "/workspace/Revisit-Anything/models/segment-anything/sam_vit_h_4b8939.pth"
    fastsam_checkpoint: str = "/workspace/Revisit-Anything/models/segment-anything/FastSAM-x.pt"
    dino_checkpoint: str = "/workspace/Revisit-Anything/models/dinov2/dinov2_vitg14_pretrain.pth"
    dino_model_name: str = "dinov2_vitg14"

    mask_segmentation: str = "fastsam"
    # global_method_name: Literal["SegLoc", "VLAD", "PCA"] = "SegLoc"
    global_method_name: str = "SegLoc"
    pca_fit: bool = True

    vlad_config: VLADConfig = VLADConfig()
    pca_config: PCAConfig = PCAConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    top_n: int = 10
    seg_topk: int = 200
