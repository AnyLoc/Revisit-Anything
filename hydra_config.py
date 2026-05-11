from typing import Literal, Optional
from dataclasses import dataclass, field

@dataclass
class Config:
    dataset_base_dir: str = "/tmp/datasets/"
    dataset_name: str = "current"
    scene_name: str = "0"

    desired_width: int = 640
    desired_height: int = 320
    sam_checkpoint: str = "/home/ubuntu/.cache/segment-anything/sam_vit_h_4b8939.pth"
