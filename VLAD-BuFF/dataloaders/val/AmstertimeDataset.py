from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors


DATASET_ROOT = "./VPR-codebase/VPR-datasets-downloader/datasets/amstertime/images/test/"
GT_ROOT = (
    "./datasets/" # BECAREFUL, this is the ground truth that comes with GSV-Cities
)

path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception(
        f"Please make sure the path {DATASET_ROOT} to amstertime dataset is correct"
    )

if not path_obj.joinpath("database") or not path_obj.joinpath("queries"):
    raise Exception(
        f"Please make sure the directories query and ref are situated in the directory {DATASET_ROOT}"
    )


class AmstertimeDataset(Dataset):
    def __init__(self, input_transform=None, positive_dist_threshold=25):
        self.input_transform = input_transform
        self.dataset_root = DATASET_ROOT

        # reference images names
        self.dbImages = np.load(GT_ROOT + "amstertime/amstertime_dbImages.npy")

        # query images names
        self.qImages = np.load(GT_ROOT + "amstertime/amstertime_qImages.npy")

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))

        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        self.database_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in self.dbImages]
        ).astype(float)
        self.queries_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in self.qImages]
        ).astype(float)

        # ground truth
        self.ground_truth = self.get_positives(positive_dist_threshold)

    def __getitem__(self, index):
        img = Image.open(self.dataset_root + self.images[index]).convert("RGB")

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def get_positives(self, rad):
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        positives = knn.radius_neighbors(
            self.queries_utms, radius=rad, return_distance=False
        )
        return positives
