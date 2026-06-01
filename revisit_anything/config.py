from typing import Literal, Optional
from dataclasses import dataclass, field


@dataclass
class RuntimeConfig:
    device: str = "auto"              # auto | cpu | cuda | cuda:N
    precision: str = "fp32"           # fp32 | fp16
    faiss_backend: str = "cpu"        # cpu | gpu
    pca_backend: str = "torch_svd"      # sklearn | torch_svd
    # 0 = auto-detect via revisit_anything.autobatch (profiles DINO+FastSAM
    # peak memory and picks the largest B that fits free*fraction of GPU).
    # Any positive integer is used as-is.
    fused_batch_size: int = 0
    h5_writer_queue: int = 16
    dataloader_num_workers: int = 2

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

    # Enables the fused DINO + FastSAM + PCA-fit GPU stage. Disable to fall
    # back to the legacy three-stage path (FeatureExtractor + PcaFitter).
    fused_pipeline: bool = True
    # When True (and fused_pipeline is enabled), per-image VLADs are persisted
    # to vlads_{W}.h5 so compute_all_vlads can skip the SegVLAD recompute and
    # only apply PCA.
    cache_all_vlads: bool = True

    vlad_config: VLADConfig = VLADConfig()
    pca_config: PCAConfig = PCAConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    top_n: int = 10
    seg_topk: int = 200
