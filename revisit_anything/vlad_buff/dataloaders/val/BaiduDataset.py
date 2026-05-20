from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation

DATASET_ROOT = (
    "./VPR-codebase/VPR-datasets-downloader/datasets/baidu/images/test/"
)
GT_ROOT = (
    "./datasets/" # BECAREFUL, this is the ground truth that comes with GSV-Cities
)

path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception(
        f"Please make sure the path {DATASET_ROOT} to baidu dataset is correct"
    )

if not path_obj.joinpath("database") or not path_obj.joinpath("queries"):
    raise Exception(
        f"Please make sure the directories queries and database are situated in the directory {DATASET_ROOT}"
    )


def get_cop_pose(file):
    """
    Takes in input of .camera file for baidu and outputs the cop numpy array [x y z] and 3x3 rotation matrix
    """
    with open(file) as f:
        lines = f.readlines()
        xyz_cop_line = lines[-2]
        # print(cop_line)
        xyz_cop = np.fromstring(xyz_cop_line, dtype=float, sep=" ")

        r1 = np.fromstring(lines[4], dtype=float, sep=" ")
        r2 = np.fromstring(lines[5], dtype=float, sep=" ")
        r3 = np.fromstring(lines[6], dtype=float, sep=" ")
        r = Rotation.from_matrix(np.array([r1, r2, r3]))
        # print(R)

        R_euler = r.as_euler("zyx", degrees=True)

    return xyz_cop, R_euler


class BaiduDataset(Dataset):
    def __init__(
        self,
        input_transform=None,
        use_ang_positives=False,
        positive_dist_threshold=10,
        ang_thresh=20,
    ):
        self.input_transform = input_transform
        self.dataset_root = DATASET_ROOT

        # reference images names
        self.dbImages = np.load(GT_ROOT + "baidu/baidu_dbImages.npy")
        self.dbImages_gt = np.load(GT_ROOT + "baidu/baidu_dbgImages_gt.npy")

        # query images names
        self.qImages = np.load(GT_ROOT + "baidu/baidu_qImages.npy")
        self.qImages_gt = np.load(GT_ROOT + "baidu/baidu_qImages_gt.npy")

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))

        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        self.ang_thresh = ang_thresh
        self.positive_dist_threshold = positive_dist_threshold

        # form pose array from db_gt .camera files
        self.db_gt_arr = np.zeros((self.num_references, 3))  # for xyz
        self.db_gt_arr_euler = np.zeros((self.num_references, 3))  # for euler angles

        for idx, db_gt_file_rel in enumerate(self.dbImages_gt):
            db_gt_file = DATASET_ROOT + db_gt_file_rel
            with open(db_gt_file) as f:
                cop_pose, cop_R = get_cop_pose(db_gt_file)
            self.db_gt_arr[idx, :] = cop_pose
            self.db_gt_arr_euler[idx, :] = cop_R

        # form pose array from q_gt .camera files
        self.q_gt_arr = np.zeros((self.num_queries, 3))  # for xyz
        self.q_gt_arr_euler = np.zeros((self.num_queries, 3))  # for euler angles

        for idx, q_gt_file_rel in enumerate(self.qImages_gt):
            q_gt_file = DATASET_ROOT + q_gt_file_rel
            with open(q_gt_file) as f:
                cop_pose, cop_R = get_cop_pose(q_gt_file)

            self.q_gt_arr[idx, :] = cop_pose
            self.q_gt_arr_euler[idx, :] = cop_R

        if use_ang_positives:
            # Find soft_positives_per_query, which are within val_positive_dist_threshold and ang_threshold
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_gt_arr)
            self.dist, self.soft_dist_positives_per_query = knn.radius_neighbors(
                self.q_gt_arr, radius=self.positive_dist_threshold, return_distance=True
            )

            # also apply the angular distance threshold
            self.soft_positives_per_query = []

            for i in range(len(self.q_gt_arr)):  # iterate over all q_gt_array
                self.ang_dist = []
                for j in range(
                    len(self.soft_dist_positives_per_query[i])
                ):  # iterate over all positive queries
                    # print(self.q_gt_arr - self.db_gt_arr[self.soft_positives_per_query[i][j]])
                    ang_diff = np.mean(
                        np.abs(
                            self.q_gt_arr_euler[i]
                            - self.db_gt_arr_euler[
                                self.soft_dist_positives_per_query[i][j]
                            ]
                        )
                    )
                    if ang_diff < self.ang_thresh:
                        self.ang_dist.append(self.soft_dist_positives_per_query[i][j])
                self.soft_positives_per_query.append(self.ang_dist)

            # Shallow MLP Training Database
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_gt_arr)
            self.dist, self.soft_dist_positives_per_db = knn.radius_neighbors(
                self.db_gt_arr,
                radius=self.positive_dist_threshold,
                return_distance=True,
            )

            self.soft_positives_per_db = []

            for i in range(len(self.db_gt_arr)):  # iterate over all q_gt_array
                self.ang_dist = []
                for j in range(
                    len(self.soft_dist_positives_per_db[i])
                ):  # iterate over all positive queries
                    # print(self.q_gt_arr - self.db_gt_arr[self.soft_positives_per_db[i][j]])
                    ang_diff = np.mean(
                        np.abs(
                            self.q_gt_arr_euler[i]
                            - self.db_gt_arr_euler[
                                self.soft_dist_positives_per_db[i][j]
                            ]
                        )
                    )
                    if ang_diff < self.ang_thresh:
                        self.ang_dist.append(self.soft_dist_positives_per_db[i][j])
                self.soft_positives_per_db.append(self.ang_dist)

        else:
            # Find soft_positives_per_query, which are within val_positive_dist_threshold only
            # self.db_gt_arr = self.db_gt_arr.tolist()
            # self.q_gt_arr = self.q_gt_arr.tolist()
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_gt_arr)
            self.dist, self.soft_positives_per_query = knn.radius_neighbors(
                self.q_gt_arr, radius=self.positive_dist_threshold, return_distance=True
            )

            # Shallow MLP Training for database
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_gt_arr)
            self.dist, self.soft_positives_per_db = knn.radius_neighbors(
                self.db_gt_arr,
                radius=self.positive_dist_threshold,
                return_distance=True,
            )

        self.database_utms, self.queries_utms = None, None
        # ground truth
        self.ground_truth = self.get_positives()

    def __getitem__(self, index):
        img = Image.open(self.dataset_root + self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def get_positives(self, rad=None):
        if rad is None or self.database_utms is None:
            return self.soft_positives_per_query
