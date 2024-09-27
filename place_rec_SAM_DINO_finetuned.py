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
import func
from place_rec_global_config import datasets, experiments

cmap = matplotlib.cm.get_cmap("jet")

if __name__=="__main__":
    DINONV_extraction = True #False # # SegVLAD finetuned
    DINOSALAD_extraction = False # SALAD
    print(f"DINONV_extraction: {DINONV_extraction}, DINOSALAD_extraction: {DINOSALAD_extraction}")

    # Be careful: Even in cases when SAM_extraction is True, you may want to use full resolution. Depends on what the actual image resolution is. If it's too huge, you want to half it, else use full resolution.
    # mask_full_resolution = False #DINO always full resolution



    parser = argparse.ArgumentParser(description='SAM/DINO/FastSAM extraction for Any Dataset. See place_rec_global_config.py to see how to give arguments.')
    parser.add_argument('--dataset', required=True, help='Dataset name') # baidu, pitts etc
    parser.add_argument('--debug', action='store_true', help='Not being implemented yet.')
    args = parser.parse_args()

    # Load dataset and experiment configurations
    dataset_config = datasets.get(args.dataset, {})
    if not dataset_config:
        raise ValueError(f"Dataset '{args.dataset}' not found in configuration.")


    print(dataset_config)

    # cfg = {'rmin':0, 'desired_width':640, 'desired_height':480} # Note for later: Not using this cfg anywhere in local code currently. Should incorporate as part of local matching later.
    cfg = dataset_config['cfg']
    # mask width and height: half if mask_full_resolution is False, else True
    width_DINO, height_DINO =  cfg['desired_width'], cfg['desired_height']
    print(f"IMPORTANT: The dimensions being used for DINO extraction are both {width_DINO}x{height_DINO} pixels.")

    # if args.dataset == "pitts" or args.dataset.startswith("msls") or args.dataset == "tokyo247":
    workdir = f'/scratch/saishubodh/segments_data/{args.dataset}/out'
    os.makedirs(workdir, exist_ok=True)
    workdir_data = '/scratch/saishubodh/segments_data'
    # else: 
    #     workdir = f'/ssd_scratch/saishubodh/segments_data/{args.dataset}/out'
    #     os.makedirs(workdir, exist_ok=True)
    #     workdir_data = '/ssd_scratch/saishubodh/segments_data'
    save_path_results = f"{workdir}/results/"

    ims_sidx, ims_eidx, ims_step = 0, None, 1

    dataPath1_r = f"{workdir_data}/{args.dataset}/{dataset_config['data_subpath1_r']}/"
    dataPath2_q = f"{workdir_data}/{args.dataset}/{dataset_config['data_subpath2_q']}/"

    if DINONV_extraction:
        dino_nv_checkpoint = f"{workdir_data}/SegmentsMap_data/models/DnV2_NV/last.ckpt"
        # sam_checkpoint = f"{workdir_data}/SegmentsMap_data/models/FastSAM/FastSAM-x.pt"
        list_all = [
            {"dataPath": dataPath1_r, "h5FullPathDINO": f"{workdir}/{args.dataset}_r_dinoNV_{width_DINO}.h5"},
            {"dataPath": dataPath2_q, "h5FullPathDINO": f"{workdir}/{args.dataset}_q_dinoNV_{width_DINO}.h5"}]
    elif DINOSALAD_extraction:
        dino_salad_checkpoint = f"{workdir_data}/SegmentsMap_data/models/dino_salad.ckpt"
        # sam_checkpoint = f"{workdir_data}/SegmentsMap_data/models/segment-anything/sam_vit_h_4b8939.pth"
        list_all = [
            {"dataPath": dataPath1_r, "h5FullPathDINO": f"{workdir}/{args.dataset}_r_dinoSALAD_{width_DINO}.h5"},
            {"dataPath": dataPath2_q, "h5FullPathDINO": f"{workdir}/{args.dataset}_q_dinoSALAD_{width_DINO}.h5"} ]
    # EXTRACTION STARTS:

    if DINONV_extraction:

        for iter_dict in list_all:
        # skip r and only do q
        # for iter_dict in list_all[1:]:


            dataPath = iter_dict["dataPath"]
            ims = natsorted(os.listdir(f'{dataPath}'))
            ims = ims[ims_sidx:ims_eidx][::ims_step]
        
            h5FullPathDINONV = iter_dict["h5FullPathDINO"]
            cfg_dino = { "desired_width": width_DINO, "desired_height": height_DINO, "detect": 'dino', "use_sam": True, "class_threshold": 0.9, \
                "desired_feature": 0, "query_type": 'text', "sort_by": 'area', "use_16bit": False, "use_cuda": True,\
                        "dino_strides": 4, "use_traced_model": False, 
                        "rmin":0, "DAStoreFull":False, "dinov2": True, "wrap":False, "resize": True} # robohop specifc params
        
            print("DINONV extraction started...")
            # dino = func.loadDINO(cfg_dino, device="cuda")
            # func.process_dino_ft_to_h5(h5FullPathDINONV,cfg_dino,ims,dino,dataDir=dataPath)
            backbone = func.loadDINONV(cfg_dino, dino_nv_checkpoint,device="cuda",feat_type="backbone")
            func.process_DINONV(backbone,ims,cfg_dino,h5FullPathDINONV,dataPath)
            del backbone 

            print(f"\n \n DINONV EXTRACTED DONE at path: {h5FullPathDINONV} \n \n ")
        # print("\n \n DINONV EXTRACTED DONE \n \n NEXT DINOSALAD \n \n")
    
    if DINOSALAD_extraction:

        # only r but not q
        # for iter_dict in list_all[:1]:
        for iter_dict in list_all:
            dataPath = iter_dict["dataPath"]
            ims = natsorted(os.listdir(f'{dataPath}'))
            ims = ims[ims_sidx:ims_eidx][::ims_step]
        
            h5FullPathDINOSALAD = iter_dict["h5FullPathDINO"]
            cfg_dino = { "desired_width": width_DINO, "desired_height": height_DINO, "detect": 'dino', "use_sam": True, "class_threshold": 0.9, \
                "desired_feature": 0, "query_type": 'text', "sort_by": 'area', "use_16bit": False, "use_cuda": True,\
                        "dino_strides": 4, "use_traced_model": False, 
                        "rmin":0, "DAStoreFull":False, "dinov2": True, "wrap":False, "resize": True} # robohop specifc params
        
            print("DINO extraction started...")
            dino = func.loadDINOSALAD(cfg_dino, ckpt_path=dino_salad_checkpoint, device="cuda", feat_type="full")
            func.process_dino_salad_ft_to_h5(h5FullPathDINOSALAD,cfg_dino,ims,dino,dataDir=dataPath, device="cuda", feat_type="full", feat_return='f')
            del dino 

        print("\n \n DINOSALAD EXTRACTED DONE \n \n ")

