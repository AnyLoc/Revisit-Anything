import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from hydra_config import Config

from utilities import VLAD
import func_vpr
from loguru import logger
from natsort import natsorted
from pathlib import Path
import h5py
import numpy as np
import torch
import cv2
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm
from place_rec_sfm_pairs import compute_seg_vlad
from fastsam import FastSAM, FastSAMPrompt
import faiss
import json

import pickle


cs = ConfigStore.instance()
cs.store(name="default", node=Config)

def feature_extraction(cfg, output_folder, image_folder, ims):
    image_feature_dict_folder = output_folder/"image_feature_dicts"
    image_feature_dict_folder.mkdir(parents=True, exist_ok=True)

    h5FullPathDINO = image_feature_dict_folder / f"dino_{cfg.desired_width}.h5"
    h5FullPathMask = image_feature_dict_folder / f"masks_{cfg.desired_width}.h5"

    if h5FullPathDINO.is_file() and h5FullPathMask.is_file():
        logger.info(f"Feature H5 files already exist at {h5FullPathDINO} and {h5FullPathMask}. Skipping feature extraction.")
        return h5FullPathDINO, h5FullPathMask

    # dino extraction
    logger.info("Starting DINO extraction")
    cfg_dino = { "desired_width": cfg.desired_width, "desired_height": cfg.desired_height, "detect": 'dino', "use_sam": True, "class_threshold": 0.9, \
                "desired_feature": 0, "query_type": 'text', "sort_by": 'area', "use_16bit": False, "use_cuda": True,\
                "dino_strides": 4, "use_traced_model": False, 
                "rmin":0, "DAStoreFull":False, "dinov2": True, "wrap":False, "resize": True} # robohop specifc params

    dino_model = func_vpr.loadDINO(cfg_dino, cfg.dino_model, device="cuda")
    func_vpr.process_dino_ft_to_h5(h5FullPathDINO,cfg_dino,ims,dino_model,dataDir=str(image_folder))
    logger.info("DINO extraction completed.")

    # sam extraction
    if cfg.mask_segmentation == "sam":
        logger.info("Starting SAM extraction")
        cfg_sam = { "desired_width": cfg.desired_width, "desired_height": cfg.desired_height, "detect": 'dino', "use_sam": True, "class_threshold": 0.9, \
                    "desired_feature": 0, "query_type": 'text', "sort_by": 'area', "use_16bit": False, "use_cuda": True,\
                    "dino_strides": 4, "use_traced_model": False, 
                    "rmin":0, "DAStoreFull":False, "dinov2": True, "wrap":False, "resize": True} # robohop specifc params

        SAM = func_vpr.loadSAM(cfg.sam_checkpoint, cfg_sam, device="cuda")
        func_vpr.process_SAM_to_h5(h5FullPathMask, cfg_sam,ims,SAM)
        logger.info("SAM extraction completed.")
    elif cfg.mask_segmentation == "fastsam":
        logger.info("Starting FastSAM extraction")
        cfg_fastsam = { "desired_width": cfg.desired_width, "desired_height": cfg.desired_height, "detect": 'dino', "use_sam": True, "class_threshold": 0.9, \
                    "desired_feature": 0, "query_type": 'text', "sort_by": 'area', "use_16bit": False, "use_cuda": True,\
                    "dino_strides": 4, "use_traced_model": False,
                    "rmin":0, "DAStoreFull":False, "dinov2": True, "wrap":False, "resize": True} # robohop specifc params

        fastsam_model = func_vpr.loadSAM_FastSAM(cfg.fastsam_checkpoint, cfg_fastsam, device="cuda")
        ims_names = [im.name for im in ims]
        func_vpr.process_SAM_to_h5_FastSAM(h5FullPathMask, cfg_fastsam, ims_names, fastsam_model, device="cuda", dataDir=str(image_folder))
        logger.info("FastSAM extraction completed.")
    else:
        logger.error(f"Unknown mask_segmentation method: {cfg.mask_segmentation}. Supported methods are 'sam' and 'fastsam'.")
        raise ValueError(f"Unknown mask_segmentation method: {cfg.mask_segmentation}. Supported methods are 'sam' and 'fastsam'.")

    return h5FullPathDINO, h5FullPathMask

def fit_pca_model(h5fullPathDINO, h5FullPathSAM, cfg, output_folder, ims):
    logger.info("Starting PCA model fitting with the following configuration:")
    pca_model_path = output_folder / f"pca_model_{cfg.vlad_config.domain}.pkl"
    if pca_model_path.is_file():
        logger.info(f"PCA model already exists at {pca_model_path}. Loading PCA model from file.")
        return pca_model_path

    dino_h5 = h5py.File(str(h5fullPathDINO), "r")
    masks_h5 = h5py.File(str(h5FullPathSAM), "r")

    # setting depends on vlad training, use indoor for pretrained
    ext_specifier = f"dinov2_vitg14/l{cfg.vlad_config.desc_layer}_{cfg.vlad_config.desc_facet}_c{cfg.vlad_config.number_clusters}"
    c_centers_file = Path(cfg.vlad_config.cache_dir)/"vocabulary"/ext_specifier/cfg.vlad_config.domain / "c_centers.pt"
    if not c_centers_file.is_file():
        logger.error(f"Cluster centers not found: {c_centers_file}")
        raise FileNotFoundError(f"Cluster centers not found: {c_centers_file}")
    c_centers = torch.load(c_centers_file)
    if c_centers.shape[0] != cfg.vlad_config.number_clusters:
        logger.error(f"Number of clusters in c_centers ({c_centers.shape[0]}) does not match cfg.vlad_number_clusters ({cfg.vlad_number_clusters})")
        raise ValueError(f"Number of clusters in c_centers ({c_centers.shape[0]}) does not match cfg.vlad_number_clusters ({cfg.vlad_number_clusters})")

    vlad = VLAD(cfg.vlad_config.number_clusters, desc_dim=None, cache_dir=c_centers_file.parent)
    vlad.fit(None)

    dh = cfg.desired_height // 14
    dw = cfg.desired_width // 14
    idx_matrix = np.empty((cfg.desired_height, cfg.desired_width, 2), dtype="int32")
    for i in range(cfg.desired_height):
        for j in range(cfg.desired_width):
            idx_matrix[i, j] = np.array([np.clip(i // 14, 0, dh - 1), np.clip(j // 14, 0, dw - 1)])
    ind_matrix = np.ravel_multi_index(idx_matrix.reshape(-1, 2).T, (dh, dw))
    ind_matrix = torch.tensor(ind_matrix, device="cuda")

    logger.info("counting masks")
    numSegments = func_vpr.countNumMasksInDataset(ims, masks_h5)

    accumulated_segments = 0
    global_sampling_ratio = min(1, cfg.pca_config.max_segments/ numSegments)
    pca = PCA(n_components=cfg.pca_config.lower_dim, whiten=cfg.pca_config.whiten, svd_solver=cfg.pca_config.svd_solver)

    segFtVLAD_list = [] 
    imInds = np.array([], dtype=int)

    logger.info("Computing SegLoc for all images in the dataset...")
    for r_id, r_img in tqdm(enumerate(ims), total=len(ims), desc="Processing images..."):
        # Preload all masks for the image
        masks_seg = func_vpr.preload_masks(masks_h5, r_img.name)
        imInds_ind, regInds_ind, segMask_ind = func_vpr.getIdxSingleFast(r_id,masks_seg,minArea=cfg.pca_config.min_area)
        imInds = np.concatenate((imInds, imInds_ind))

        if cfg.pca_config.order:
            adjMat1_ind = func_vpr.nbrMasksAGGFastSingle(masks_seg, cfg.pca_config.order)
        else:
            adjMat1_ind = None

        gd = func_vpr.seg_vlad_gpu_single(ind_matrix, idx_matrix, dino_h5, r_img.name, segMask_ind, c_centers, cfg, desc_dim=cfg.vlad_config.desc_dim, adj_mat=adjMat1_ind)

        gd = gd.to(dtype=torch.float32) # Convert to float32 for PCA to keep RAM in check

        current_batch_size = gd.shape[0]
        sample_size = int(current_batch_size * global_sampling_ratio)
        
        if cfg.pca_fit:
            if sample_size > 0:
                sample_indices = torch.randperm(current_batch_size)[:sample_size]
                sampled_gd = gd[sample_indices]
                segFtVLAD_list.append(sampled_gd)
                accumulated_segments += sampled_gd.shape[0]
        else:
            segFtVLAD_list.append(gd) #imfts_batch same as gd here, in the full image function, it is for 100 images at a time
        
        if accumulated_segments >= cfg.pca_config.max_segments:
            break
    segFtVLAD = torch.cat(segFtVLAD_list, dim=0)
    del segFtVLAD_list
    if cfg.pca_fit:
        logger.info(f"Fitting PCA model on {accumulated_segments} segments (sampled from {numSegments} total segments) with the following PCA configuration:")
        pca.fit(segFtVLAD.numpy())
        logger.info(f"Explained variance ratio of PCA components: {pca.explained_variance_ratio_}")
        logger.info("Saving PCA model to pickle file...")
        # elif args.vocab_vlad == 'map':
            # pca_model_path = f"{workdir}/{args.dataset}{experiment_config['pca_model_pkl_map']}"
        # else:
            # raise ValueError(f"Unknown vocab-vlad value: {args.vocab_vlad}")

        with open(pca_model_path, "wb") as file:
            pickle.dump(pca, file)
        logger.info(f"PCA model saved to {pca_model_path}")
    return pca_model_path

def create_pairs(pca_model_path, ims, h5fullPathDINO, h5FullPathSAM, cfg, output_folder):
    logger.info("Start Creating pairs")
    ext_specifier = f"dinov2_vitg14/l{cfg.vlad_config.desc_layer}_{cfg.vlad_config.desc_facet}_c{cfg.vlad_config.number_clusters}"
    c_centers_file = Path(cfg.vlad_config.cache_dir)/"vocabulary"/ext_specifier/cfg.vlad_config.domain / "c_centers.pt"
    if not c_centers_file.is_file():
        logger.error(f"Cluster centers not found: {c_centers_file}")
        raise FileNotFoundError(f"Cluster centers not found: {c_centers_file}")
    c_centers = torch.load(c_centers_file)
    if c_centers.shape[0] != cfg.vlad_config.number_clusters:
        logger.error(f"Number of clusters in c_centers ({c_centers.shape[0]}) does not match cfg.vlad_number_clusters ({cfg.vlad_number_clusters})")
        raise ValueError(f"Number of clusters in c_centers ({c_centers.shape[0]}) does not match cfg.vlad_number_clusters ({cfg.vlad_number_clusters})")

    dh = cfg.desired_height // 14
    dw = cfg.desired_width // 14
    idx_matrix = np.empty((cfg.desired_height, cfg.desired_width, 2), dtype="int32")
    for i in range(cfg.desired_height):
        for j in range(cfg.desired_width):
            idx_matrix[i, j] = np.array([np.clip(i // 14, 0, dh - 1), np.clip(j // 14, 0, dw - 1)])
    ind_matrix = np.ravel_multi_index(idx_matrix.reshape(-1, 2).T, (dh, dw))
    ind_matrix = torch.tensor(ind_matrix, device="cuda")

    feats, im_inds = compute_seg_vlad(
        ims, h5fullPathDINO, h5FullPathSAM, c_centers, cfg,
        ind_matrix, idx_matrix,
        order=cfg.pca_config.order if cfg.pca_fit else 0,
        pca_model_path=pca_model_path,
        min_area=cfg.pca_config.min_area,
    )

    feats_np = feats.numpy()
    if pca_model_path is not None:
        feats_np = func_vpr.normalizeFeat(feats_np)
    index = faiss.IndexFlatL2(feats_np.shape[1])
    index.add(feats_np)
    sims, matches = index.search(feats_np, cfg.seg_topk)
    logger.info(f"faiss search: sims={sims.shape}, matches={matches.shape}")

    # 3) Aggregate per-image top-(N+1) using the existing borda method,
    #    then strip self.
    sims_50 = 2 - sims[:, :50]
    matches_50 = matches[:, :50]
    seg_range = [np.where(im_inds == i)[0] for i in range(len(ims))]
    dummy_gt = [[] for _ in range(len(ims))]
    preds = func_vpr.get_matches(
        matches_50, dummy_gt, sims_50, seg_range, im_inds,
        n=cfg.top_n+ 1, method="max_seg_topk_wt_borda_Im",
    )

    pair_map = {}
    pair_set = set()
    for i, img in enumerate(ims):
        nbrs = [int(j) for j in preds[i] if int(j) != i][:cfg.top_n]
        pair_map[img.name] = [ims[j].name for j in nbrs]
        for j in nbrs:
            a, b = sorted([img.name, ims[j].name])
            pair_set.add((a, b))

    # 4) Write outputs
    json_path = output_folder/ f"pairs_top{cfg.top_n}.json"
    with open(str(json_path), "w") as f:
        json.dump(pair_map, f, indent=2)
    logger.info(f"Wrote {json_path}")

    txt_path = output_folder / f"pairs_top{cfg.top_n}.txt"
    with open(str(txt_path), "w") as f:
        for a, b in sorted(pair_set):
            f.write(f"{a} {b}\n")
    logger.info(f"Wrote {txt_path}  ({len(pair_set)} unique pairs)")

@hydra.main(config_name="default")
def main(cfg: DictConfig):
    cfg = OmegaConf.structured(cfg)
    workdir = Path(cfg.dataset_base_dir)/cfg.dataset_name/cfg.scene_name
    output_folder = workdir/f"outputs"
    image_folder = workdir/"images"

    # TODO start end step for ims
    ims_sidx, ims_eidx, ims_step = 0, None, 1
    ims = natsorted(list(image_folder.glob("*")))
    ims = ims[ims_sidx:ims_eidx][::ims_step]
    n_imgs = len(ims)
    if n_imgs <= 2:
        logger.warning(f"Number of images to process is {n_imgs}. Please check if the image folder {image_folder} is correct and contains the expected number of images.")
        raise ValueError(f"Number of images to process is {n_imgs}. Please check if the image folder {image_folder} is correct and contains the expected number of images.")
    if cfg.top_n >= n_imgs:
        logger.warning(f"top_n is set to {cfg.top_n}, which is greater than or equal to the number of images ({n_imgs}). This may lead to unexpected behavior. Please set top_n to a value less than {n_imgs}.")
        raise ValueError(f"top_n is set to {cfg.top_n}, which is greater than or equal to the number of images ({n_imgs}). Please set top_n to a value less than {n_imgs}.")


    h5fullPathDINO, h5FullPathSAM = feature_extraction(cfg, output_folder, image_folder, ims)
    if cfg.global_method_name == "SegLoc":
        pca_model_path = fit_pca_model(h5fullPathDINO, h5FullPathSAM, cfg, output_folder, ims)
    else:
        raise NotImplementedError(f"Global method {cfg.global_method_name} not implemented in this script. Only 'SegLoc' is implemented.")

    create_pairs(pca_model_path, ims, h5fullPathDINO, h5FullPathSAM, cfg, output_folder)

if __name__ == "__main__":
    main()
