
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

from dataloaders.baidu_dataloader import Baidu_Dataset
from dataloaders.vpair_dataloader import VPAir
from dataloaders.MapillaryDatasetVal import MSLS 

def get_gt(dataset, cfg, workdir_data, ims1_r=None, ims2_q=None, func_vpr_module=None):
    """
    Retrieves the ground truth (gt) based on the specified dataset.

    Parameters:
        dataset (str): The name of the dataset.
        cfg (dict): Configuration settings.
        workdir_data (str): Path to the working directory data.
        ims1_r (list, optional): List of reference image paths (required for some datasets).
        ims2_q (list, optional): List of query image paths (required for some datasets).
        func_vpr_module (module, optional): Module containing VPR-related functions.

    Returns:
        gt: Ground truth data structure appropriate for the dataset.
    """
    if dataset == "baidu":
        vpr_dl = Baidu_Dataset(cfg, workdir_data, 'baidu') 
        gt = vpr_dl.soft_positives_per_query

    elif dataset in ["mslsSF", "mslsCPH"]:
        GT_ROOT = './dataloaders/msls_npy_files/'
        city_name = "sf" if dataset == "mslsSF" else "cph"
        vpr_dl = MSLS(city_name=city_name, GT_ROOT=GT_ROOT)
        gt = vpr_dl.soft_positives_per_query

    elif dataset == "pitts":
        npy_pitts_path = f"{workdir_data}/{dataset}/pitts30k/images/test/"
        db = np.load(f"{npy_pitts_path}database.npy")
        q = np.load(f"{npy_pitts_path}queries.npy")
        utmDb = func_vpr_module.get_utm(db)
        utmQ = func_vpr_module.get_utm(q)
        gt = func_vpr_module.get_positives(utmDb, utmQ, 25)

    elif dataset == "SFXL":
        if ims1_r is None or ims2_q is None:
            raise ValueError("ims1_r and ims2_q must be provided for the SFXL dataset.")
        database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in ims1_r]).astype(float)
        queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in ims2_q]).astype(float)
        positive_dist_threshold = 25
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(database_utms)
        gt = knn.radius_neighbors(queries_utms, radius=positive_dist_threshold, return_distance=False)

    elif dataset == "InsideOut":
        utmDb_path = f"{workdir_data}/{dataset}/gps_db_correct.pkl"
        utmQ_path = f"{workdir_data}/{dataset}/gps_q_new.pkl"
        utmDb = pickle.load(open(utmDb_path, "rb"))
        utmQ = pickle.load(open(utmQ_path, "rb"))
        gt = func_vpr_module.get_positives(utmDb, utmQ, 50)

    elif dataset == "17places":
        if ims2_q is None:
            raise ValueError("ims2_q must be provided for the 17places dataset.")
        loc_rad = 15
        gt = [list(np.arange(i - loc_rad, i + loc_rad + 1)) for i in range(len(ims2_q))]

    elif dataset == "AmsterTime":
        if ims1_r is None:
            raise ValueError("ims1_r must be provided for the AmsterTime dataset.")
        gt = [[i] for i in range(len(ims1_r))]

    elif dataset == "VPAir":
        vpr_dl = VPAir(cfg, workdir_data, 'VPAir')
        gt = vpr_dl.soft_positives_per_query

    else:
        print("Dataset not found but saving descriptors, calculate recall later")
        gt = None  # Ensures descriptors are saved; recall can be calculated later.

    return gt
