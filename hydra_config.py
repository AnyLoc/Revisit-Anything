from typing import Literal, Optional
from dataclasses import dataclass, field

@dataclass
class Config:
    dataset_base_dir: str = "/home/ubuntu/"
    dataset_name: str = "search-data"
    scene_name: str = "0"

    desired_width: int = 640
    desired_height: int = 320
    sam_checkpoint: str = "/home/ubuntu/search-data/models/segment-anything/sam_vit_h_4b8939.pth"
    dino_checkpoint: str = "/home/ubuntu/search-data/models/DINO/dinov2_vitg14/dinov2_vitg14.pth"
