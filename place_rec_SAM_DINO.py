import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
from natsort import natsorted
import networkx as nx
import h5py
from glob import glob
import pickle
from importlib import reload
from tqdm import tqdm
import numpy as np

import argparse
import func_vpr
from place_rec_global_config import datasets, workdir_data


def set_extraction_method(method):
    if method == 'DINO':
        return True, False, False
    elif method == 'SAM':
        return False, True, False
    elif method == 'FastSAM':
        return False, False, True
    else:
        raise ValueError("Invalid method. Choose from 'DINO', 'SAM', or 'FastSAM'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SAM/DINO/FastSAM extraction for Any Dataset. See place_rec_global_config.py to see how to give arguments.')
    parser.add_argument('--dataset', required=True, help='Dataset name')  # baidu, pitts etc
    parser.add_argument('--method', required=True, choices=['DINO', 'SAM'],
                        help="Choose extraction method: 'DINO', 'SAM'")

    args = parser.parse_args()

    DINO_extraction, SAM_extraction, FastSAM_extraction = set_extraction_method(args.method)

    mask_full_resolution = False  # DINO always full resolution

    # Load dataset and experiment configurations
    dataset_config = datasets.get(args.dataset, {})
    if not dataset_config:
        raise ValueError(f"Dataset '{args.dataset}' not found in configuration.")

    print(dataset_config)

    cfg = dataset_config['cfg']
    # mask width and height: half if mask_full_resolution is False, else True
    if mask_full_resolution:  # DINO always full resolution
        width_SAM, height_SAM = cfg['desired_width'], cfg['desired_height']
        width_DINO, height_DINO = cfg['desired_width'], cfg['desired_height']
        print(f"Note: The dimensions being used for SAM and DINO extraction are both {width_SAM}x{height_SAM} pixels.")
    else:
        if args.dataset == "AmsterTime":  # use full resolution for AmsterTime as original res is itself small
            width_SAM, height_SAM = cfg['desired_width'], cfg['desired_height']
            width_DINO, height_DINO = cfg['desired_width'], cfg['desired_height']
            print(
                f"Note: The dimensions being used for SAM and DINO extraction are both {width_SAM}x{height_SAM} pixels.")
        else:
            width_SAM, height_SAM = int(0.5 * cfg['desired_width']), int(0.5 * cfg['desired_height'])
            width_DINO, height_DINO = cfg['desired_width'], cfg['desired_height']
            print(
                f"Note: The dimensions being used for SAM extraction are {width_SAM}x{height_SAM} pixels and for DINO extraction are {width_DINO}x{height_DINO} pixels.")

    workdir = f'{workdir_data}/{args.dataset}/out'
    os.makedirs(workdir, exist_ok=True)
    save_path_results = f"{workdir}/results/"

    ims_sidx, ims_eidx, ims_step = 0, None, 1

    dataPath1_r = f"{workdir_data}/{args.dataset}/{dataset_config['data_subpath1_r']}/"
    dataPath2_q = f"{workdir_data}/{args.dataset}/{dataset_config['data_subpath2_q']}/"

    if FastSAM_extraction:
        sam_checkpoint = f"{workdir_data}/models/FastSAM/FastSAM-x.pt"
        list_all = [
            {"dataPath": dataPath1_r, "h5FullPathDINO": f"{workdir}/{args.dataset}_r_dino_{width_DINO}.h5",
             "h5FullPathSAM": f"{workdir}/{args.dataset}_r_FastSAM_masks_{width_SAM}.h5"},
            {"dataPath": dataPath2_q, "h5FullPathDINO": f"{workdir}/{args.dataset}_q_dino_{width_DINO}.h5",
             "h5FullPathSAM": f"{workdir}/{args.dataset}_q_FastSAM_masks_{width_SAM}.h5"}]
    else:
        sam_checkpoint = f"{workdir_data}/models/segment-anything/sam_vit_h_4b8939.pth"
        list_all = [
            {"dataPath": dataPath1_r, "h5FullPathDINO": f"{workdir}/{args.dataset}_r_dino_{width_DINO}.h5",
             "h5FullPathSAM": f"{workdir}/{args.dataset}_r_masks_{width_SAM}.h5"},
            {"dataPath": dataPath2_q, "h5FullPathDINO": f"{workdir}/{args.dataset}_q_dino_{width_DINO}.h5",
             "h5FullPathSAM": f"{workdir}/{args.dataset}_q_masks_{width_SAM}.h5"}]

    if FastSAM_extraction:
        raise NotImplementedError("FastSAM extraction is not implemented in this script. To be updated soon.")
    #     for iter_dict in list_all:
    #         dataPath = iter_dict["dataPath"]
    #         ims = natsorted(os.listdir(f'{dataPath}'))
    #         ims = ims[ims_sidx:ims_eidx][::ims_step]

    #         h5FullPathSAM = iter_dict["h5FullPathSAM"]
    #         cfg_sam = { "desired_width": width_SAM, "desired_height": height_SAM, "detect": 'dino', "use_sam": True, "class_threshold": 0.9, \
    #             "desired_feature": 0, "query_type": 'text', "sort_by": 'area', "use_16bit": False, "use_cuda": True,\
    #                     "dino_strides": 4, "use_traced_model": False, 
    #                     "rmin":0, "DAStoreFull":False, "dinov2": True, "wrap":False, "resize": True} 

    #         print("FastSAM extraction started...")
    #         FastSAM = func_vpr.loadSAM_FastSAM(sam_checkpoint,cfg_sam, device="cuda")
    #         func_vpr.process_SAM_to_h5_FastSAM(h5FullPathSAM, cfg_sam,ims,FastSAM,dataDir=dataPath)
    #         print("\n \n FastSAM EXTRACTED DONE \n \n")

    if DINO_extraction:

        cfg_dino = {"desired_width": width_DINO, "desired_height": height_DINO, "detect": 'dino', "use_sam": True,
                    "class_threshold": 0.9,
                    "desired_feature": 0, "query_type": 'text', "sort_by": 'area', "use_16bit": False,
                    "use_cuda": True,
                    "dino_strides": 4, "use_traced_model": False,
                    "rmin": 0, "DAStoreFull": False, "dinov2": True, "wrap": False,
                    "resize": True}  # robohop specifc params

        print("DINO extraction started...")
        dino = func_vpr.loadDINO(cfg_dino, device="cuda")

        for iter_dict in list_all:
            dataPath = iter_dict["dataPath"]
            ims = natsorted(os.listdir(f'{dataPath}'))
            ims = ims[ims_sidx:ims_eidx][::ims_step]

            h5FullPathDINO = iter_dict["h5FullPathDINO"]
            func_vpr.process_dino_ft_to_h5(h5FullPathDINO, cfg_dino, ims, dino, dataDir=dataPath)

        print("\n \n DINO EXTRACTED DONE \n \n ")

    if SAM_extraction:
        cfg_sam = {"desired_width": width_SAM, "desired_height": height_SAM, "detect": 'dino', "use_sam": True,
                   "class_threshold": 0.9,
                   "desired_feature": 0, "query_type": 'text', "sort_by": 'area', "use_16bit": False,
                   "use_cuda": True,
                   "dino_strides": 4, "use_traced_model": False,
                   "rmin": 0, "DAStoreFull": False, "dinov2": True, "wrap": False,
                   "resize": True}  # robohop specifc params

        print("SAM extraction started...")
        SAM = func_vpr.loadSAM(sam_checkpoint, cfg_sam, device="cuda")

        for iter_dict in list_all:
            dataPath = iter_dict["dataPath"]
            ims = natsorted(os.listdir(f'{dataPath}'))
            ims = ims[ims_sidx:ims_eidx][::ims_step]

            h5FullPathSAM = iter_dict["h5FullPathSAM"]
            func_vpr.process_SAM_to_h5(h5FullPathSAM, cfg_sam, ims, SAM, dataDir=dataPath)
            # sanity check for resizing
            # f = h5py.File(h5FullPathSAM,'r')
            # keys = list(f.keys())
            # print(f[keys[0]]['masks']['21']['segmentation'])

        print("\n \n SAM EXTRACTED DONE \n \n ")
