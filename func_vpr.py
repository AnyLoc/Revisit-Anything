# import func, func_sr
import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from dataloaders.baidu_dataloader import Baidu_Dataset
from dataloaders.aerial_dataloader import Aerial
#from dataloaders.datasets_vg import map_builder
#from dataloaders.datasets_vg import util
#import utm
from glob import glob
from collections import defaultdict
import os
from os.path import join
from natsort import natsorted
import cv2
from typing import Literal, List
import torch
from tkinter import *
import matplotlib
from utilities import VLAD
from sklearn.decomposition import PCA
import pickle
import faiss
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from torchvision import transforms as tvf

from sam.segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from utilities import DinoV2ExtractFeatures
from DINO.dino_wrapper import get_dino_pixel_wise_features_model, preprocess_frame

# from FastSAM.fastsam import FastSAM, FastSAMPrompt 
# from FastSAM.utils.tools import convert_box_xywh_to_xyxy

import torch.nn.functional as F
# matplotlib.use('TkAgg')
workdir = '/media/kartik/data/kartik/data/segrec/out'
workdir_data = '/media/kartik/data/kartik/data/segrec/'
# cfg = {'rmin':0, 'desired_width':640, 'desired_height':480}

def first_k_unique_indices(ranked_indices, K):
    """
    Obtain the first K unique indices from a ranked list of N indices.

    :param ranked_indices: List[int] - List of ranked indices
    :param K: int - Number of unique indices to obtain
    :return: List[int] - List containing first K unique indices
    """
    seen = set()
    return [x for x in ranked_indices if x not in seen and (seen.add(x) or True)][:K]

def weighted_borda_count(*ranked_lists_with_scores):
    """
    Merge ranked lists using a weighted Borda Count method where each index's score
    is based on its similarity score rather than its position.
    
    :param ranked_lists_with_scores: Variable number of tuples/lists containing (index, score) pairs.
    :return: A list of indices sorted by their aggregated scores.
    """
    scores = {}
    for ranked_list in ranked_lists_with_scores:
        for index, score in ranked_list:
            if index in scores:
                scores[index] += score
            else:
                scores[index] = score
    sorted_indices = sorted(scores.keys(), key=lambda index: scores[index], reverse=True)
    return sorted_indices


def get_matches(matches,gt,sims,segRangeQuery,imIndsRef,n=1,method="max_sim"):
    """
    final version seems to be max_seg_topk_wt_borda_Im (need to confirm)
    
    """
    preds=[]
    for i in range(len(gt)):
        if method == "max_sim":
            match = np.flip(np.argsort(sims[segRangeQuery[i]])[-50:])
            # pred_match.append(match)
            match_patch = matches[segRangeQuery[i]][match]
            pred = imIndsRef[match_patch]
            pred_top_k = first_k_unique_indices(pred,n)
            preds.append(pred_top_k)
        elif method == "max_seg":
            match_patch = matches[segRangeQuery[i]]
            segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
            pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-n:])]
            # sim_score_t = sim_img.T[i][pred]
            
            # preds.append(pred[np.flip(np.argsort(sim_score_t))])
            preds.append(pred)
        elif method =="max_seg_sim":
            match_patch = matches[segRangeQuery[i]]
            segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
            pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-6:])]
            sims_patch = sims[segRangeQuery[i]]
            sim_temp=[]
            for j in range(len(pred)):
                try:
                    sim_temp.append(np.max(sims_patch[np.where(imIndsRef[match_patch]==pred[j])[0]]))
                except:
                     print("index: ", i)
                     print("pred: ", pred[j])
                     print("imInds: ", imIndsRef[match_patch])
            pred = pred[np.flip(np.argsort(sim_temp))][:n]
            preds.append(pred)

        elif method=="max_seg_topk":
            match_patch = matches[segRangeQuery[i]].flatten()
            segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
            pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-n:])]
            # sim_score_t = sim_img.T[i][pred]
            
            # preds.append(pred[np.flip(np.argsort(sim_score_t))])
            preds.append(pred)
        elif method=="max_seg_topk_borda":
            match_patch = matches[segRangeQuery[i]].T.tolist()
            match_patch = merge_ranked_lists(*match_patch)
            segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
            pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-n:])]
            # sim_score_t = sim_img.T[i][pred]
            
            # preds.append(pred[np.flip(np.argsort(sim_score_t))])
            preds.append(pred)
        elif method=="max_seg_topk_avg":
            match_patch = matches[segRangeQuery[i]].T.tolist()
            match_patch = average_rank_method(*match_patch)
            segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
            pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-n:])]
            # sim_score_t = sim_img.T[i][pred]
            
            # preds.append(pred[np.flip(np.argsort(sim_score_t))])
            preds.append(pred)
        elif method=="max_seg_topk_wt_borda":
            match_patch = matches[segRangeQuery[i]].T.tolist()
            #TODO min max norm
            # sims_patch = sims[segRangeQuery[i]].T.tolist()
            sims_patch = sims[segRangeQuery[i]].T
            sims_max = np.max(sims)
            sims_min = np.min(sims)
            sims_patch = (sims_patch - sims_min)/(sims_max-sims_min)
            sims_patch = sims_patch.tolist()
            pair_patch = [list(zip(match_patch[k],sims_patch[k])) for k in range(len(sims_patch))]
            # pair_patch = [list(zip(match_patch[k],sims_patch[k])) for k in range(len(sims_patch))]
            match_patch = weighted_borda_count(*pair_patch)
            segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
            pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-n:])]
            # sim_score_t = sim_img.T[i][pred]
            
            # preds.append(pred[np.flip(np.argsort(sim_score_t))])
            preds.append(pred)
        elif method=="max_seg_topk_avg_sim":
            match_patch = matches[segRangeQuery[i]].T.tolist()
            #TODO min max norm
            sims_patch = sims[segRangeQuery[i]].T
            sims_max = np.max(sims)
            sims_min = np.min(sims)
            sims_patch = (sims_patch - sims_min)/(sims_max-sims_min)
            sims_patch = sims_patch.tolist()
            pair_patch = [list(zip(match_patch[k],sims_patch[k])) for k in range(len(sims_patch))]
            # print(sims_patch)
            # break
            match_patch = average_similarity_scores(*pair_patch)
            segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
            pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-n:])]
            # sim_score_t = sim_img.T[i][pred]
            
            # preds.append(pred[np.flip(np.argsort(sim_score_t))])
            preds.append(pred)
        #using imInds
        elif method=="max_seg_topk":
            match_patch = matches[segRangeQuery[i]].flatten()
            segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
            pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-n:])]
            # sim_score_t = sim_img.T[i][pred]
            
            # preds.append(pred[np.flip(np.argsort(sim_score_t))])
            preds.append(pred)
        elif method=="max_seg_topk_borda_Im":
            match_patch = matches[segRangeQuery[i]].T.tolist()
            match_patch = merge_ranked_lists(*imIndsRef[match_patch])
            # segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
            # pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-n:])]
            # sim_score_t = sim_img.T[i][pred]
            
            # preds.append(pred[np.flip(np.argsort(sim_score_t))])
            preds.append(match_patch[:n])
        elif method=="max_seg_topk_avg_Im":
            match_patch = matches[segRangeQuery[i]].T.tolist()
            match_patch = average_rank_method(*imIndsRef[match_patch])
            # segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
            # pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-n:])]
            # sim_score_t = sim_img.T[i][pred]
            
            # preds.append(pred[np.flip(np.argsort(sim_score_t))])
            preds.append(match_patch[:n])
        elif method=="max_seg_topk_wt_borda_Im":
            match_patch = matches[segRangeQuery[i]].T.tolist()
            #TODO min max norm
            # sims_patch = sims[segRangeQuery[i]].T.tolist()
            sims_patch = sims[segRangeQuery[i]].T
            sims_max = np.max(sims)
            sims_min = np.min(sims)
            sims_patch = (sims_patch - sims_min)/(sims_max-sims_min)
            sims_patch = sims_patch.tolist()
            pair_patch = [list(zip(imIndsRef[match_patch[k]],sims_patch[k])) for k in range(len(sims_patch))]
            # pair_patch = [list(zip(match_patch[k],sims_patch[k])) for k in range(len(sims_patch))]
            match_patch = weighted_borda_count(*pair_patch)
            segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
            pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-n:])]
            # sim_score_t = sim_img.T[i][pred]
            
            # preds.append(pred[np.flip(np.argsort(sim_score_t))])
            preds.append(match_patch[:n])
        elif method=="max_seg_topk_avg_sim_Im":
            match_patch = matches[segRangeQuery[i]].T.tolist()
            #TODO min max norm
            sims_patch = sims[segRangeQuery[i]].T
            sims_max = np.max(sims)
            sims_min = np.min(sims)
            sims_patch = (sims_patch - sims_min)/(sims_max-sims_min)
            sims_patch = sims_patch.tolist()
            pair_patch = [list(zip(imIndsRef[match_patch[k]],sims_patch[k])) for k in range(len(sims_patch))]
            # print(sims_patch)
            # break
            match_patch = average_similarity_scores(*pair_patch)
            segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
            pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-n:])]
            # sim_score_t = sim_img.T[i][pred]
            
            # preds.append(pred[np.flip(np.argsort(sim_score_t))])
            preds.append(match_patch[:n])
    return preds



def get_matches_for_single_image_pair(matches,sims,segRangeQuery,imIndsRef,n=1,method="max_sim"):
    """
    Here, we are only considering a single image pair for the query and reference images,
    rather than the entire dataset. This is for qualitative analysis purposes.

    Using max_sim currently for this analysis as that seems to make sense for a given image pair. 
    Need to think about borda soon

    Although final version for full dataset testing is be max_seg_topk_wt_borda_Im .
    """
    preds=[]
    i = 0
    # for i in range(len(gt)):
    if method == "max_sim":
        match = np.flip(np.argsort(sims[segRangeQuery[i]])[-50:])
        # pred_match.append(match)
        sorted_query_segment_indices = match
        sorted_reference_image_indices = matches[segRangeQuery[i]][match]
        # match_patch = matches[segRangeQuery[i]][match]

        # pred = imIndsRef[match_patch]
        # pred_top_k = first_k_unique_indices(pred,n)
        # preds.append(pred_top_k)
    return sorted_query_segment_indices, sorted_reference_image_indices
    # elif method == "max_seg":
    #     match_patch = matches[segRangeQuery[i]]
    #     segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
    #     pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-n:])]
    #     # sim_score_t = sim_img.T[i][pred]
        
    #     # preds.append(pred[np.flip(np.argsort(sim_score_t))])
    #     preds.append(pred)
    # elif method =="max_seg_sim":
    #     match_patch = matches[segRangeQuery[i]]
    #     segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
    #     pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-6:])]
    #     sims_patch = sims[segRangeQuery[i]]
    #     sim_temp=[]
    #     for j in range(len(pred)):
    #         try:
    #             sim_temp.append(np.max(sims_patch[np.where(imIndsRef[match_patch]==pred[j])[0]]))
    #         except:
    #                 print("index: ", i)
    #                 print("pred: ", pred[j])
    #                 print("imInds: ", imIndsRef[match_patch])
    #     pred = pred[np.flip(np.argsort(sim_temp))][:n]
    #     preds.append(pred)

    # elif method=="max_seg_topk_wt_borda_Im":
    #     match_patch = matches[segRangeQuery[i]].T.tolist()
    #     #TODO min max norm
    #     # sims_patch = sims[segRangeQuery[i]].T.tolist()
    #     sims_patch = sims[segRangeQuery[i]].T
    #     sims_max = np.max(sims)
    #     sims_min = np.min(sims)
    #     sims_patch = (sims_patch - sims_min)/(sims_max-sims_min)
    #     sims_patch = sims_patch.tolist()
    #     pair_patch = [list(zip(imIndsRef[match_patch[k]],sims_patch[k])) for k in range(len(sims_patch))]
    #     # pair_patch = [list(zip(match_patch[k],sims_patch[k])) for k in range(len(sims_patch))]
    #     match_patch = weighted_borda_count(*pair_patch)
    #     segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
    #     pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-n:])]
    #     # sim_score_t = sim_img.T[i][pred]
        
    #     # preds.append(pred[np.flip(np.argsort(sim_score_t))])
    #     preds.append(match_patch[:n])
    # return preds


def get_matches_old(matches,gt,sims,segRangeQuery,imIndsRef,n=1,method="max_sim"):
    preds=[]
    for i in range(len(gt)):
        if method == "max_sim":
            match = np.flip(np.argsort(sims[segRangeQuery[i]])[-50:])
            # pred_match.append(match)
            match_patch = matches[segRangeQuery[i]][match]
            pred = imIndsRef[match_patch]
            pred_top_k = first_k_unique_indices(pred,n)
            preds.append(pred_top_k)
        elif method == "max_seg":
            match_patch = matches[segRangeQuery[i]]
            segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
            pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-n:])]
            # sim_score_t = sim_img.T[i][pred]
            
            # preds.append(pred[np.flip(np.argsort(sim_score_t))])
            preds.append(pred)
        elif method =="max_seg_sim":
            match_patch = matches[segRangeQuery[i]]
            segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
            pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-6:])]
            sims_patch = sims[segRangeQuery[i]]
            sim_temp=[]
            for j in range(len(pred)):
                try:
                    sim_temp.append(np.max(sims_patch[np.where(imIndsRef[match_patch]==pred[j])[0]]))
                except:
                     print("index: ", i)
                     print("pred: ", pred[j])
                     print("imInds: ", imIndsRef[match_patch])
            pred = pred[np.flip(np.argsort(sim_temp))][:n]
            preds.append(pred)
    return preds


def convert_to_queries_results_for_map(max_seg_preds, gt):
    queries_results = []
    for query_idx, refs in enumerate(max_seg_preds):
        query_results = [ref in gt[query_idx] for ref in refs]
        queries_results.append(query_results)
    return queries_results


def calculate_ap(retrieved_items):
    """
    Calculate the average precision (AP) for a single query.
    retrieved_items: a list of boolean values, where True indicates a relevant item, and False indicates a non-relevant item.
    """
    relevant_items = sum(retrieved_items)
    if relevant_items == 0:
        return 0  # Return 0 if there are no relevant items
    cumsum = 0
    precision_at_k = 0
    for i, is_relevant in enumerate(retrieved_items, start=1):
        if is_relevant:
            cumsum += 1
            precision_at_k += cumsum / i
    average_precision = precision_at_k / relevant_items
    return average_precision

def calculate_map(queries_results):
    """
    Calculate the mean average precision (mAP) for a set of queries.
    queries_results: a list of lists, where each inner list represents the retrieved items for a query,
    with True for relevant items and False for non-relevant items.
    # Example usage
    queries_results = [
        [True, False, True, False, True],  # Query 1: list of retrieved items (True = relevant, False = not relevant)
        [False, True, True, False, False], # Query 2
        [True, True, False, False, False], # Query 3
    ]
    map_score = calculate_map(queries_results)
    print(f"Mean Average Precision (mAP): {map_score}")
    """
    ap_scores = [calculate_ap(query) for query in queries_results]
    return sum(ap_scores) / len(ap_scores) if ap_scores else 0



def calc_recall(pred,gt,n,analysis=False):
    recall=[0]*n
    recall_per_query=[0]*len(gt)
    num_eval = 0
    for i in range(len(gt)):
        if len(gt[i])==0:
                continue
        num_eval+=1
        for j in range(len(pred[i])):
            # print(len(max_seg_preds[i]))
            # print(i)

            if n==1:
                if pred[i] in gt[i]:
                    recall[j]+=1
                    recall_per_query[i]=1
                    break
            else:
                if pred[i][j] in gt[i]:
                    recall[j]+=1
                    break

    recalls = np.cumsum(recall)/float(num_eval)
    print("POSITIVES/TOTAL segVLAD for this dataset: ", np.cumsum(recall),"/", num_eval)
    if analysis:
        return recalls.tolist(), recall_per_query
    return recalls.tolist()

def unpickle(file):
    pickle_out = open(file,'rb')
    desc = pickle.load(pickle_out)
    pickle_out.close()
    return desc

def getIdxs(ims,masks_in, minArea=400, retunrMask = True):
        imInds =[]
        regInds=[]
        segMasks =[]
        for i in tqdm(range(len(ims))):
                im_name = ims[i]
                key =f"{im_name}"
                # print(key)
                segRange=[]
                regIndsIm=[]
                segmask=[]
                count = 0
                for k in natsorted(masks_in[key+'/masks/'].keys()):           
                    mask = masks_in[key+f'/masks/{k}/']

                # for mask in masks:
                    # m = mask['segmentation'][()]

                        # m = torch.nn.functional.interpolate(m.float().unsqueeze(0).unsqueeze(0), [cfg['desired_height'],cfg['desired_width']], mode = 'nearest').squeeze().bool()
                    if mask['area'][()]>minArea:
                        if retunrMask:
                            segmask.append(mask['segmentation'][()])
                        regIndsIm.append(count)
                        imInds.append(i)
                        count+=1
                regInds.append(regIndsIm)
                segMasks.append(segmask)
        return np.array(imInds), regInds, segMasks

def getIdxs_simple_without_segMasks(ims,masks_in, minArea=400, retunrMask = True):
        imInds =[]
        regInds=[]
        segMasks =[]
        for i in tqdm(range(len(ims))):
                im_name = ims[i]
                key =f"{im_name}"
                # print(key)
                segRange=[]
                regIndsIm=[]
                # segmask=[]
                count = 0
                for k in natsorted(masks_in[key+'/masks/'].keys()):           
                    # mask = masks_in[key+f'/masks/{k}/']

                # for mask in masks:
                    # m = mask['segmentation'][()]

                        # m = torch.nn.functional.interpolate(m.float().unsqueeze(0).unsqueeze(0), [cfg['desired_height'],cfg['desired_width']], mode = 'nearest').squeeze().bool()
                    # if mask['area'][()]>minArea:
                    #     if retunrMask:
                    #         segmask.append(mask['segmentation'][()])
                    regIndsIm.append(count)
                    imInds.append(i)
                    # print(imInds)
                    count+=1
                regInds.append(regIndsIm)
                # segMasks.append(segmask)
        return np.array(imInds), regInds, segMasks 

def getAnyLocFt(img, extractor, device='cuda',upsample=True):
    base_tf = tvf.Compose([tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])])
    with torch.no_grad():
        img_pt = base_tf(img).to(device)
        # Make image patchable (14, 14 patches)
        c, h, w = img_pt.shape
        h_reduced, w_reduced = h // 14, w // 14 # h_r * w_r = 17 * 22 = 374
        h_new, w_new = h_reduced * 14, w_reduced * 14
        img_pt = tvf.CenterCrop((h_new, w_new))(img_pt)[None, ...]
        # Extract descriptor
        feat = extractor(img_pt) # [1, num_patches, desc_dim] i.e. [1, 374, 1536]
        feat = feat.reshape(1,h_reduced,w_reduced,-1) # [1, 17, 22, 1536]
        feat = feat.permute(0, 3, 1, 2) # [1, 1536, 17, 22]
        if upsample:
            feat = torch.nn.functional.interpolate(feat, [h,w], mode="bilinear", align_corners=True) # [1, 1536, 240, 320]
    return feat #pixel wise features: [1, 1536, 240, 320] for img size of (240, 320, 3) if upsample=True else [1, 1536, 17, 22]



def loadSAM(sam_checkpoint, cfg, device = 'cuda'):
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    return mask_generator

def loadSAM_FastSAM(fastsam_checkpoint, cfg, device = 'cuda'):
    model = FastSAM(fastsam_checkpoint)
    model.to(device)  # Ensure model is moved to the appropriate device

    # model_type = "vit_t"
    # sam = mobile_sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)
    # mask_generator = MobileSamAutomaticMaskGenerator(sam)

    return model 


def loadDINO(cfg, device = "cuda"):
    if cfg['dinov2']:
        dino = DinoV2ExtractFeatures("dinov2_vitg14", 31, 'value', device='cuda',norm_descs=False)
    else:
        dino = get_dino_pixel_wise_features_model(cfg = cfg, device = device)    

    return dino

def process_single_SAM(cfg, img, models, device):
    mask_generator = models
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if cfg['resize']:
        img_p = cv2.resize(img, (cfg['desired_width'],cfg['desired_height']))
 
    else : img_p = img
    masks = mask_generator.generate(img_p)

    return img_p, masks

def process_single_DINO(cfg,img,models,device):
    dino = models
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if cfg['resize']:
        img_p = cv2.resize(img, (cfg['desired_width'],cfg['desired_height']))
    else : img_p =img
    if cfg['dinov2']:
        img_feat = getAnyLocFt(img_p, dino, device, upsample=False)
        # img_feat = getAnyLocFt(img_p, dino, device, upsample=True) #shub
    else:
        img_d = preprocess_frame(img_p, cfg=cfg)
        img_feat = dino.forward(img_d)
    img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)
    return img_p, img_feat_norm

def masks_given_image(SAM, ims_i, dataPath1, cfg, mask_full_resolution=False, device="cuda"):
    rmin = cfg['rmin']

    if mask_full_resolution: #DINO always full resolution
        width_SAM, height_SAM =  cfg['desired_width'], cfg['desired_height']
        print(f"IMPORTANT: The dimensions being used for SAM  extraction are  {width_SAM}x{height_SAM} pixels.")
    else:
        width_SAM, height_SAM =  int(0.5 * cfg['desired_width']), int(0.5 * cfg['desired_height'])
        print(f"IMPORTANT: The dimensions being used for SAM  extraction are  {width_SAM}x{height_SAM} pixels.")


    masks_seg = []
    im = cv2.imread(f'{dataPath1}/{ims_i}')[rmin:,:,:]
    cfg_sam = { "desired_width": width_SAM, "desired_height": height_SAM, "detect": 'dino', "use_sam": True, "class_threshold": 0.9, \
        "desired_feature": 0, "query_type": 'text', "sort_by": 'area', "use_16bit": False, "use_cuda": True,\
                "dino_strides": 4, "use_traced_model": False, 
                "rmin":0, "DAStoreFull":False, "dinov2": True, "wrap":False, "resize": True} # robohop specifc params
    im_p, masks = process_single_SAM(cfg_sam, im, SAM, device)
    for j, m in enumerate(masks):
        # for k in m.keys():
        # import pdb; pdb.set_trace()
        mask_area = m['area']
        # if mask_area < 12000:
        # if mask_area < 7000:
        #     masks_seg.append(m['segmentation'])#[()])
        masks_seg.append(m['segmentation'])#[()])

    return masks_seg, masks


def masks_given_image_old(sam_checkpoint, ims_i, dataPath1, cfg, mask_full_resolution=False, device="cuda"):
    rmin = cfg['rmin']

    if mask_full_resolution: #DINO always full resolution
        width_SAM, height_SAM =  cfg['desired_width'], cfg['desired_height']
        print(f"IMPORTANT: The dimensions being used for SAM  extraction are  {width_SAM}x{height_SAM} pixels.")
    else:
        width_SAM, height_SAM =  int(0.5 * cfg['desired_width']), int(0.5 * cfg['desired_height'])
        print(f"IMPORTANT: The dimensions being used for SAM  extraction are  {width_SAM}x{height_SAM} pixels.")

    cfg_sam = { "desired_width": width_SAM, "desired_height": height_SAM, "detect": 'dino', "use_sam": True, "class_threshold": 0.9, \
        "desired_feature": 0, "query_type": 'text', "sort_by": 'area', "use_16bit": False, "use_cuda": True,\
                "dino_strides": 4, "use_traced_model": False, 
                "rmin":0, "DAStoreFull":False, "dinov2": True, "wrap":False, "resize": True} # robohop specifc params

    SAM = loadSAM(sam_checkpoint,cfg_sam, device="cuda")

    masks_seg = []
    im = cv2.imread(f'{dataPath1}/{ims_i}')[rmin:,:,:]
    im_p, masks = process_single_SAM(cfg_sam, im, SAM, device)
    for j, m in enumerate(masks):
        # for k in m.keys():
        # import pdb; pdb.set_trace()
        mask_area = m['area']
        # if mask_area < 12000:
        if mask_area < 7000:
            masks_seg.append(m['segmentation'])#[()])
        # masks_seg.append(m['segmentation'])#[()])

    return masks_seg


def dino_given_image(dino, ims_i, dataPath1, cfg, device="cuda"):
    rmin = cfg['rmin']

    width_DINO, height_DINO =  cfg['desired_width'], cfg['desired_height']

    cfg_dino = { "desired_width": width_DINO, "desired_height": height_DINO, "detect": 'dino', "use_sam": True, "class_threshold": 0.9, \
        "desired_feature": 0, "query_type": 'text', "sort_by": 'area', "use_16bit": False, "use_cuda": True,\
                "dino_strides": 4, "use_traced_model": False, 
                "rmin":0, "DAStoreFull":False, "dinov2": True, "wrap":False, "resize": True} # robohop specifc params


    im = cv2.imread(f'{dataPath1}/{ims_i}')[rmin:,:,:]
    im_p, ift_dino = process_single_DINO(cfg_dino,im,dino,device)
    # ift_dino = ift_dino.detach().cpu().numpy()

    # dino_desc = torch.from_numpy(ift_dino) 
    # return dino_desc
    ift_dino = ift_dino.detach().cpu()
    return ift_dino


def process_dino_ft_to_h5(h5FullPath,cfg,ims,models,device = "cuda",dataDir="./"):
    rmin = cfg['rmin']
    with h5py.File(h5FullPath, "w") as f:
        # loop over images and create a h5py group per image
        # imnames = sorted(os.listdir(f'{workdir}/{datasetpath}'))
        # for i, imname in enumerate(imnames):
        #     im = cv2.imread(f'{workdir}/{datasetpath}/{imname}')
        for i, _ in enumerate(tqdm(ims)):
            if isinstance(ims[0],str):
                imname = ims[i] 
                im = cv2.imread(f'{dataDir}/{imname}')[rmin:,:,:]
            else:
                imname, im = i, ims[i][rmin:,:,:]
            im_p, ift_dino = process_single_DINO(cfg,im,models,device)
            grp = f.create_group(f"{imname}")
            grp.create_dataset("ift_dino", data=ift_dino.detach().cpu().numpy())

def process_SAM_to_h5(h5FullPath,cfg,ims,models,device="cuda",dataDir="./"):
    rmin = cfg['rmin']
    with h5py.File(h5FullPath, "w") as f:
        for i, _ in enumerate(tqdm(ims)):
            if isinstance(ims[0],str):
                imname = ims[i] 
                im = cv2.imread(f'{dataDir}/{imname}')[rmin:,:,:]
            else:
                imname, im = i, ims[i][rmin:,:,:]
            im_p, masks = process_single_SAM(cfg,im,models,device)
            grp = f.create_group(f"{imname}")
            grp.create_group("masks")
            for j, m in enumerate(masks):
                for k in m.keys():
                    grp["masks"].create_dataset(f"{j}/{k}", data=m[k])    


def process_SAM_to_h5_FastSAM(h5FullPath,cfg,ims,model,device="cuda",dataDir="./"):
    rmin = cfg['rmin']
    with h5py.File(h5FullPath, "w") as f:
        for i, _ in enumerate(tqdm(ims)):
            if isinstance(ims[0],str):
                imname = ims[i] 
                # im = cv2.imread(f'{dataDir}/{imname}')[rmin:,:,:]
                im = Image.open(f'{dataDir}/{imname}')
            else:
                imname, im = i, ims[i][rmin:,:,:]
            masks = process_single_FastSAM(cfg,im,model,imname, device)
            grp = f.create_group(f"{imname}")
            grp.create_group("masks")
            for j, m in enumerate(masks):
                for k in m.keys():
                    grp["masks"].create_dataset(f"{j}/{k}", data=m[k])    

def process_single_FastSAM(cfg,input_image,model, imname, device="cuda", imgsz=1024, iou=0.9, conf=0.4, retina=True):
    
    input_image = input_image.convert("RGB")

    if cfg['resize']:
        img_np = np.array(input_image)
        img_np = cv2.resize(img_np, (cfg['desired_width'], cfg['desired_height']))
        input_image = Image.fromarray(img_np)
    
    everything_results = model(
        input_image,
        device=device,
        retina_masks=retina,
        imgsz=imgsz,
        conf=conf,
        iou=iou    
    )

    prompt_process = FastSAMPrompt(input_image, everything_results, device=device)
    ann = prompt_process.everything_prompt()
    # ann is # Tensor of shape torch.Size([num_objects, height, width])
    if isinstance(ann, torch.Tensor):
        ann_numpy = ann.cpu().numpy().astype(bool)
    else: #if it is an empty list
        print("Empty list - no masks found for image name : ", imname)
        print("printing ann for verification: ", ann)

        # create a mask of two types: one with all ones and one with one random pixel set to True
        ann_numpy = np.zeros((2, cfg['desired_height'], cfg['desired_width']), dtype=bool) # careful: numpy uses oppsoite convention of opencv for h,w

        ann_numpy[0] = np.ones((cfg['desired_height'], cfg['desired_width']), dtype=bool) # careful: numpy uses oppsoite convention of opencv for h,w
        # ann_numpy = np.zeros((1, cfg['desired_height'], cfg['desired_width']), dtype=bool)
        
        random_row = np.random.randint(0, cfg['desired_height'])
        random_col = np.random.randint(0, cfg['desired_width'])
        
        # Set the randomly chosen pixel to True
        ann_numpy[1, random_row, random_col] = True



    # Create the list of dictionaries with segmentation masks
    masks = [{"segmentation": ann_numpy[i]} for i in range(ann_numpy.shape[0])]


    return masks  


def preload_masks(masks_in, image_key):
    """
    Preloads all masks for a given image key from the HDF5 file.
    
    Args:
    - masks_in: An open h5py File or Group object representing the HDF5 file or a group within it.
    - image_key: The key in the HDF5 file for the specific image.
    
    Returns:
    - A list of mask data loaded into memory.
    """
    masks_path = f"{image_key}/masks/"
    mask_keys = natsorted(masks_in[masks_path].keys())
    masks_seg = [masks_in[masks_path + k]['segmentation'][()] for k in mask_keys]
    return masks_seg

def getIdxSingleFast(img_idx, masks_seg, minArea=400, returnMask=True):
    imInds = []
    regIndsIm = []
    segmask = []
    count = 0
    
    # im_name = ims[img_idx]
    # key = f"{im_name}"
    
    # Preload all masks for the image
    # masks_seg, mask_keys = preload_masks(masks_in, key)
    
    for mask in masks_seg:
        # Assuming 'area' is a property that can be efficiently accessed without loading the entire mask
        # This may require adjusting based on your actual HDF5 structure and how 'area' is stored
        # area = masks_in[f"{key}/masks/{k}/area"][()]
        
        # if area > minArea:
        if returnMask:
            segmask.append(mask)
        regIndsIm.append(count)
        imInds.append(img_idx)
        count += 1
    
    return np.array(imInds), regIndsIm, segmask

def countNumMasksInDataset(ims, masks_in):
    count = 0
    for im_name in tqdm(ims, desc="Counting num of masks in dataset"):
        # Directly constructing the path to the masks for the current image
        mask_path = f"{im_name}/masks/"
        if mask_path in masks_in:
            # Assuming each key under mask_path corresponds to one mask,
            # and we can count them directly.
            mask_keys = natsorted(masks_in[mask_path].keys())
            count += len(mask_keys)
    return count


def getIdxSingleFast_for_single_image_pair(masks_seg, minArea=400, returnMask=True):
    """
    This function is for the case where we are doing say qual analysis for a single image pair of query and reference images. So this function is over all the masks of a single image rather than taking all the masks from full dataset.
    """

    img_idx = 0 # Only 1 image is handled every function call
    imInds = []
    regIndsIm = []
    segmask = []
    count = 0
    
    # im_name = ims[img_idx]
    # key = f"{im_name}"
    
    # Preload all masks for the image
    # masks_seg, mask_keys = preload_masks(masks_in, key)
    
    for mask in masks_seg:
        # Assuming 'area' is a property that can be efficiently accessed without loading the entire mask
        # This may require adjusting based on your actual HDF5 structure and how 'area' is stored
        # area = masks_in[f"{key}/masks/{k}/area"][()]
        
        # if area > minArea:
        if returnMask:
            segmask.append(mask)
        regIndsIm.append(count)
        imInds.append(img_idx)
        count += 1
    
    return np.array(imInds), regIndsIm, segmask


def get_recall(database_vectors, query_vectors, gt, analysis =False, k=5):
    # Original PointNetVLAD code
    # import pdb;pdb.set_trace()
    # if database_vectors.dtype!=np.float32:
    #     database_output = database_vectors.detach().cpu().numpy()
    #     queries_output = query_vectors.detach().cpu().numpy()
    # else: 
    database_output = database_vectors
    queries_output = query_vectors
    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = k
    recall = [0] * num_neighbors
    recall_per_query=[0]*len(queries_output)
    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)
    matches =[]
    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        # query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = gt[i]
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
        dict_info = {'seg_id_q': -1, 'img_id_r': indices[0], 'seg_id_r': -1, 'img_id_to_seg_id': -1}
        matches.append(dict_info)
        # import pdb;pdb.set_trace()
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
 
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                recall_per_query[i]=1
                break
        # import pdb;pdb.set_trace()
        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    print("POSITIVES/TOTAL AnyLoc for this dataset: ", np.cumsum(recall),"/", num_evaluated)
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    if analysis:
        return recall, recall_per_query, matches
    return recall,matches
    # import pdb;pdb.set_trace()

def aggFt(desc_path, masks, segRange, cfg,aggType, vlad = None, upsample = False, segment_global = False,segment = False):
    f = h5py.File(desc_path, "r")
    # keys = list(f.keys())
    keys = list(natsorted(f.keys()))
    imFts=[]
    for i in tqdm(range(len(keys))):
        if aggType =="avg":
            segfeat = torch.empty([1,1536,0]).to('cuda')
            dino_desc = torch.from_numpy(f[keys[i]]['ift_dino'][()]).to('cuda')
            if upsample:
                dino_desc = torch.nn.functional.interpolate(dino_desc, [cfg['desired_height'],cfg['desired_width']], mode="bilinear", align_corners=True)
            dino_desc_norm = torch.nn.functional.normalize(dino_desc, dim=1)
            if segment_global:
                for j in range(len(segRange[i])):
                    # mask = torch.from_numpy(G.nodes[segRange[i][j]]['segmentation']).to('cuda')
                    mask = masks[i][segRange[i][j]].to('cuda')
                    mask = torch.nn.functional.interpolate(mask.float().unsqueeze(0).unsqueeze(0), [cfg['desired_height'],cfg['desired_width']], mode = 'nearest').squeeze().bool()
                    reg_feat_norm = dino_desc_norm[:,:,mask]
                
                    segfeat = torch.cat((segfeat,reg_feat_norm),axis =2)
                imFt = segfeat.mean(axis =2).detach().cpu().numpy()
                imFt = np.reshape(imFt, (imFt.shape[-1],))
                imFts.append(imFt)
            if segment:
                for j in range(len(segRange[i])):
                    # mask = torch.from_numpy(G.nodes[segRange[i][j]]['segmentation']).to('cuda')
                    mask = torch.from_numpy(masks[i][j]).to('cuda')
                    if upsample:
                        mask = torch.nn.functional.interpolate(mask.float().unsqueeze(0).unsqueeze(0), [cfg['desired_height'],cfg['desired_width']], mode = 'nearest').squeeze().bool()
                    else :
                        mask = torch.nn.functional.interpolate(mask.float().unsqueeze(0).unsqueeze(0), [cfg['desired_height']//14,cfg['desired_width']//14], mode = 'nearest').squeeze().bool()

                    reg_feat_norm = dino_desc_norm[:,:,mask].mean(axis=2).detach().cpu().numpy()
                    imFt = reg_feat_norm
                    imFt = np.reshape(imFt, (imFt.shape[-1],))
                    imFts.append(imFt)
            else :
                imFt = dino_desc_norm.mean([2,3]).detach().cpu().numpy()
                imFt = np.reshape(imFt, (imFt.shape[-1],))
                imFts.append(imFt)
        elif aggType=="vlad":
            if segment:
                dino_desc = torch.from_numpy(f[keys[i]]['ift_dino'][()]).to('cuda')
                if upsample:
                    dino_desc = torch.nn.functional.interpolate(dino_desc, [cfg['desired_height'],cfg['desired_width']], mode="bilinear", align_corners=True)

                dino_desc_norm = torch.nn.functional.normalize(dino_desc, dim=1)
                for j in range(len(segRange[i])):
                    mask = torch.from_numpy(masks[i][j]).to('cuda')
                    # mask = torch.from_numpy(G.nodes[segRange[i][j]]['segmentation']).to('cuda')
                    if upsample:
                        mask = torch.nn.functional.interpolate(mask.float().unsqueeze(0).unsqueeze(0), [cfg['desired_height'],cfg['desired_width']], mode = 'nearest').squeeze().bool()
                    else :
                        mask = torch.nn.functional.interpolate(mask.float().unsqueeze(0).unsqueeze(0), [cfg['desired_height']//14,cfg['desired_width']//14], mode = 'nearest').squeeze().bool()

                    reg_feat_norm = dino_desc_norm[:,:,mask]
                    reg_feat_per = reg_feat_norm.permute(0,2,1)
                    gd = vlad.generate(reg_feat_per.cpu().squeeze())
                    gd_np = gd.numpy()
                    imFts.append(gd_np)
            # segfeat = torch.empty([1,49152,0])
            else:
                dino_desc = torch.from_numpy(np.reshape(f[keys[i]]['ift_dino'][()],(1,1536,f[keys[i]]['ift_dino'][()].shape[2]*f[keys[i]]['ift_dino'][()].shape[3]))).to('cuda')
                # if upsample:
                #     dino_desc = torch.nn.functional.interpolate(dino_desc, [cfg['desired_height'],cfg['desired_width']], mode="bilinear", align_corners=True)
                dino_desc_norm = torch.nn.functional.normalize(dino_desc, dim=1)
                dino_desc_per = dino_desc_norm.permute(0,2,1)
                gd = vlad.generate(dino_desc_per.cpu().squeeze())
                gd_np = gd.numpy()
                imFts.append(gd_np)
    return imFts

# from numba import jit
# @jit(nopython=True)
def seg_vlad(desc_path, segMask, segRange, vlad, cfg):
        f = h5py.File(desc_path, "r")
        keys = list(f.keys())
        imFts=torch.empty((0,49152))
        imfts_batch = torch.empty((0,49152))
        idx = np.empty((cfg['desired_height'],cfg['desired_width'],2)).astype('int32')
        for i in range(cfg['desired_height']):
                for j in range(cfg['desired_width']):
                        idx[i,j] = np.array([np.clip(i//14,0,cfg['desired_height']//14-1) ,np.clip(j//14,0,cfg['desired_width']//14-1)])
        i=0
        for i in tqdm(range(len(keys))):
                mask_list=[]
                dino_desc = torch.from_numpy(f[keys[i]]['ift_dino'][()]).to('cuda')
                # dino_desc = torch.nn.functional.interpolate(dino_desc, [cfg['desired_height'],cfg['desired_width']], mode="bilinear", align_corners=True)
                dino_desc = torch.from_numpy(np.reshape(f[keys[i]]['ift_dino'][()],(1,1536,f[keys[i]]['ift_dino'][()].shape[2]*f[keys[i]]['ift_dino'][()].shape[3]))).to('cuda')
                # dino_desc=torch.reshape(dino_desc,(1,1536,500*500)).to('cuda')
                dino_desc_norm = torch.nn.functional.normalize(dino_desc, dim=1)
                for j in range(len(segRange[i])):
                        mask = torch.from_numpy(segMask[i][j]).to('cuda')
                        # mask = torch.from_numpy(G.nodes[segRange[i][j]]['segmentation']).to('cuda')
                        mask = torch.nn.functional.interpolate(mask.float().unsqueeze(0).unsqueeze(0), [cfg['desired_height'],cfg['desired_width']], mode = 'nearest').squeeze().bool()
                        mask_list.append(mask.cpu())
                # reg_feat_norm = dino_desc_norm[:,:,mask]
                reg_feat_per = dino_desc_norm.permute(0,2,1)
                gd,_ = vlad.generate(reg_feat_per.cpu().squeeze(),idx,mask_list)
                # gd_np = gd.numpy()
                # imFts= np.vstack([imFts,gd_np])
                imfts_batch = torch.cat((imfts_batch,gd),dim=0)
                if i%100 ==0:
                      imFts = torch.cat((imFts,imfts_batch),dim=0)
                      imfts_batch = torch.empty((0,49152))
                # else:
        imFts = torch.cat((imFts,imfts_batch),dim=0)
        return imFts
        # segfeat = torch.empty([1,49152,0])

        # segfeat = torch.empty([1,49152,0])

def seg_vlad_gpu(desc_path, segMask, segRange, c_centers, cfg, desc_dim = 1536, adj_mat=None):
        f = h5py.File(desc_path, "r")
        keys = list(f.keys())
        vlad_dim=32*desc_dim
        imFts=torch.empty((0,vlad_dim))
        imfts_batch = torch.empty((0,vlad_dim))
        dh = cfg['desired_height']//14
        dw = cfg['desired_width']//14
        idx = np.empty((cfg['desired_height'],cfg['desired_width'],2)).astype('int32')
        for i in range(cfg['desired_height']):
                for j in range(cfg['desired_width']):
                        idx[i,j] = np.array([np.clip(i//14,0,cfg['desired_height']//14-1) ,np.clip(j//14,0,cfg['desired_width']//14-1)])
        i=0
        ind = np.ravel_multi_index(idx.reshape(-1,2).T,(cfg['desired_height']//14,cfg['desired_width']//14))
        ind = torch.tensor(ind, device = 'cuda')
        for i in tqdm(range(len(keys))):
                # mask_list=[]
                dino_desc = torch.from_numpy(f[keys[i]]['ift_dino'][()]).to('cuda')
                # dino_desc = torch.nn.functional.interpolate(dino_desc, [cfg['desired_height'],cfg['desired_width']], mode="bilinear", align_corners=True)
                dino_desc = torch.from_numpy(np.reshape(f[keys[i]]['ift_dino'][()],(1,desc_dim,f[keys[i]]['ift_dino'][()].shape[2]*f[keys[i]]['ift_dino'][()].shape[3]))).to('cuda')
                # dino_desc=torch.reshape(dino_desc,(1,1536,500*500)).to('cuda')
                dino_desc_norm = torch.nn.functional.normalize(dino_desc, dim=1)
                # t1 = time.time()
                mask = torch.from_numpy(np.array(segMask[i])).to('cuda')
                mask = torch.nn.functional.interpolate(mask.float().unsqueeze(0), [cfg['desired_height'],cfg['desired_width']], mode = 'nearest').squeeze().bool().reshape(len(segMask[i]),-1)
                mask_idx = torch.zeros((len(segMask[i]),dh*dw),device="cuda").bool()
                mask_ind_flat = torch.argwhere(mask)
                mask_idx[mask_ind_flat[:,0],ind[mask_ind_flat[:,1]]] = True
                # print(mask_idx.shape)
                # mask_list_ar = np.concatenate(segMask[i], axis = 0).reshape((len(segMask[i]),segMask[i][0].shape[0],segMask[i][0].shape[1])).astype('float64')
                # mask_tensor = torch.tensor(mask_list_ar,device="cuda")
                # mask_low = torch.nn.functional.interpolate(mask_tensor.float().unsqueeze(0), [cfg['desired_height']//14,cfg['desired_width']//14], mode = 'bilinear').squeeze().bool()
                # masks_low = mask_low.reshape(len(segMask[0]),-1)
                # for j in range(len(segRange[i])):
                #         mask = torch.from_numpy(segMask[i][j]).to('cuda')
                #         # mask = torch.from_numpy(G.nodes[segRange[i][j]]['segmentation']).to('cuda')
                #         mask = torch.nn.functional.interpolate(mask.float().unsqueeze(0).unsqueeze(0), [cfg['desired_height'],cfg['desired_width']], mode = 'nearest').squeeze().bool()
                #         idx_un = idx[mask.cpu()]
                #         mask_idx = np.zeros((35,35),dtype=bool)
                #         mask_idx[idx_un[:,0],idx_un[:,1]] = True
                #         mask_list.append(mask.cpu())
                #         # mask_list.append(mask_idx)
                # reg_feat_norm = dino_desc_norm[:,:,mask]
                reg_feat_per = dino_desc_norm.permute(0,2,1)
                # t2 = time.time()
                # print("time taken for mask prep: ", t2-t1)
                
                # return reg_feat_per,idx,mask_list
                if adj_mat is not None:
                    gd= vlad_single(reg_feat_per.squeeze(),c_centers.to('cuda'),idx,mask_idx, adj_mat[i].to("cuda"))
                else:
                     gd= vlad_single(reg_feat_per.squeeze(),c_centers.to('cuda'),idx,mask_idx)
                # t3 = time.time()
                # print("time taken by vlad: ", t3-t2)
                # gd_og, labels_og = vlad.generate(reg_feat_per.cpu().squeeze(),idx,mask_list)
                # return gd, labels, [], []
                # gd_np = gd.numpy()
                # imFts= np.vstack([imFts,gd_np])
                # print(imfts_batch)
                imfts_batch = torch.cat((imfts_batch,gd.cpu()),dim=0)
                if i%100 ==0:
                      imFts = torch.cat((imFts,imfts_batch),dim=0)
                      imfts_batch = torch.empty((0,vlad_dim))                # else:
        imFts = torch.cat((imFts,imfts_batch),dim=0)
        return imFts
        # segfeat = torc

def seg_vlad_gpu_single(ind, idx, desc_path_in, img_key, segMask, c_centers, cfg, desc_dim=1536, adj_mat=None):
    vlad_dim = 32 * desc_dim
    dh = cfg['desired_height'] // 14
    dw = cfg['desired_width'] // 14
    # idx = np.empty((cfg['desired_height'], cfg['desired_width'], 2)).astype('int32')
    # for i in range(cfg['desired_height']):
    #     for j in range(cfg['desired_width']):
    #         idx[i, j] = np.array([np.clip(i//14, 0, dh-1), np.clip(j//14, 0, dw-1)])
    # ind = np.ravel_multi_index(idx.reshape(-1, 2).T, (dh, dw))
    # ind = torch.tensor(ind, device='cuda')
    
    # dino_desc = torch.from_numpy(f[img_key]['ift_dino'][()]).to('cuda')
    # dino_desc = torch.from_numpy(np.reshape(f[img_key]['ift_dino'][()],(1,desc_dim,f[img_key]['ift_dino'][()].shape[2]*f[keys[i]]['ift_dino'][()].shape[3]))).to('cuda')

    dino_desc = torch.from_numpy(desc_path_in[img_key]['ift_dino'][()]) 
    # dino_desc = torch.nn.functional.interpolate(dino_desc, [cfg['desired_height'],cfg['desired_width']], mode="bilinear", align_corners=True)
    total_elements = dino_desc.shape[2] * dino_desc.shape[3]
    dino_desc = dino_desc.reshape(1, desc_dim, total_elements).to('cuda')


    dino_desc_norm = torch.nn.functional.normalize(dino_desc, dim=1)

    #IMPORTANT: could be wrong here
    mask = torch.from_numpy(np.array(segMask)).to('cuda')
    mask = torch.nn.functional.interpolate(mask.float().unsqueeze(0), [cfg['desired_height'],cfg['desired_width']], mode = 'nearest').squeeze().bool().reshape(len(segMask),-1)
    mask_idx = torch.zeros((len(segMask),dh*dw),device="cuda").bool()
    mask_ind_flat = torch.argwhere(mask)
    mask_idx[mask_ind_flat[:,0],ind[mask_ind_flat[:,1]]] = True

    reg_feat_per = dino_desc_norm.permute(0, 2, 1)
    if adj_mat is not None:
        gd, execution_time = vlad_single(reg_feat_per.squeeze(), c_centers.to('cuda'), idx, mask_idx, adj_mat.to("cuda"))
    else:
        gd, execution_time = vlad_single(reg_feat_per.squeeze(), c_centers.to('cuda'), idx, mask_idx)

    return gd.cpu() 
    # return gd.cpu() , execution_time

def seg_vlad_gpu_single_img(ind, idx, dino_desc, img_key, segMask, c_centers, cfg, desc_dim=1536, adj_mat=None):
    vlad_dim = 32 * desc_dim
    dh = cfg['desired_height'] // 14
    dw = cfg['desired_width'] // 14
    # idx = np.empty((cfg['desired_height'], cfg['desired_width'], 2)).astype('int32')
    # for i in range(cfg['desired_height']):
    #     for j in range(cfg['desired_width']):
    #         idx[i, j] = np.array([np.clip(i//14, 0, dh-1), np.clip(j//14, 0, dw-1)])
    # ind = np.ravel_multi_index(idx.reshape(-1, 2).T, (dh, dw))
    # ind = torch.tensor(ind, device='cuda')
    
    # dino_desc = torch.from_numpy(f[img_key]['ift_dino'][()]).to('cuda')
    # dino_desc = torch.from_numpy(np.reshape(f[img_key]['ift_dino'][()],(1,desc_dim,f[img_key]['ift_dino'][()].shape[2]*f[keys[i]]['ift_dino'][()].shape[3]))).to('cuda')

    # dino_desc = torch.from_numpy(desc_path_in[img_key]['ift_dino'][()]) 
    # dino_desc = torch.nn.functional.interpolate(dino_desc, [cfg['desired_height'],cfg['desired_width']], mode="bilinear", align_corners=True)
    total_elements = dino_desc.shape[2] * dino_desc.shape[3]
    dino_desc = dino_desc.reshape(1, desc_dim, total_elements).to('cuda')


    dino_desc_norm = torch.nn.functional.normalize(dino_desc, dim=1)

    #IMPORTANT: could be wrong here
    mask = torch.from_numpy(np.array(segMask)).to('cuda')
    mask = torch.nn.functional.interpolate(mask.float().unsqueeze(0), [cfg['desired_height'],cfg['desired_width']], mode = 'nearest').squeeze().bool().reshape(len(segMask),-1)
    mask_idx = torch.zeros((len(segMask),dh*dw),device="cuda").bool()
    mask_ind_flat = torch.argwhere(mask)
    mask_idx[mask_ind_flat[:,0],ind[mask_ind_flat[:,1]]] = True

    reg_feat_per = dino_desc_norm.permute(0, 2, 1)
    if adj_mat is not None:
        gd, execution_time = vlad_single(reg_feat_per.squeeze(), c_centers.to('cuda'), idx, mask_idx, adj_mat.to("cuda"))
    else:
        gd, execution_time = vlad_single(reg_feat_per.squeeze(), c_centers.to('cuda'), idx, mask_idx)

    return gd.cpu() 

def vlad_single(query_descs,c_centers,idx,masks,adj_mat=None):
    # desc_dim = 1536
    num_clusters =32
    # intra_norm = True
    
    c_centers_norm = F.normalize(c_centers,dim=1).to('cuda')
    labels = torch.argmax(query_descs @ c_centers_norm.T,dim=1)
    # labels=torch.linalg.norm(query_descs[:,None,:] - c_centers[None,:,:],axis=2).argmin(1)
    # print("labels_shape: ", labels.shape)
    # used_clusters = torch.unique(labels)
    #TODO: use normed clusters
    residuals = query_descs - c_centers[labels]

    # un_vlad = torch.zeros(num_clusters * desc_dim).to('cuda')
    # n_vlad_all = torch.empty((0,num_clusters * desc_dim)).to('cuda')
    # n_vlad_all=[]
    # mask_list_ar = np.concatenate(masks, axis = 0).reshape((len(masks),masks[0].shape[0],masks[0].shape[1])).astype('float64')
    # mask_tensor = torch.tensor(mask_list_ar,device="cuda")

    # pool = torch.nn.AdaptiveMaxPool2d(35)
    # mask_tensor_d = pool(mask_tensor.unsqueeze(0))
    # un_vlad = torch.zeros(num_clusters * desc_dim)
    # n_vlad_all = torch.empty((0,num_clusters * desc_dim))
    # idx_un = idx[masks[0]]
    # m = torch.nn.MaxPool2d((14, 14), stride=(14, 14), padding=0)
    # mask_tensor_d2 = m(masks.unsqueeze(0).float())
    # torch.argwhere(mask_tensor_d2[0,0])
    # masks_low = mask_tensor.reshape(len(masks),-1)
    # print (masks_low.shape)
    # print(residuals.shape)
    if adj_mat is not None:
        # n_vlad = vlad_matmuls_per_cluster(num_clusters,masks.double(),residuals.double(),labels, adjMat=adj_mat.double())
        n_vlad, execution_time = vlad_matmuls_per_cluster(num_clusters,masks.double(),residuals.double(),labels, adjMat=adj_mat.double())
    else:
        # n_vlad = vlad_matmuls_per_cluster(num_clusters,masks.double(),residuals.double(),labels)
        n_vlad, execution_time = vlad_matmuls_per_cluster(num_clusters,masks.double(),residuals.double(),labels)
    # print(np.argwhere(mask_tensor_d[0,0]))
    # print(idx_un)
    # return n_vlad
    return n_vlad, execution_time

def vlad_matmuls_per_cluster(num_c,masks,res,clus_labels,adjMat=None,device='cuda'):
    """
    Expects input tensors to be cuda and float/double
    """
    start_time = time.time()

    vlads = []
    num_m = len(masks)


    if adjMat is None:
        adjMat = torch.eye(num_m,dtype=masks.dtype,device=masks.device)
        # print(masks.device)
    # print(masks.shape)
    for li in range(num_c):
        
        inds_li = torch.where(clus_labels==li)[0].to(device)
        # print(masks[:,inds_li].shape)
        # return 0
        masks_nbrAgg = (adjMat @ masks[:,inds_li])
        vlad = masks_nbrAgg.bool().to(masks.dtype) @ res[inds_li,:]
        vlad = F.normalize(vlad, dim=1)
        vlads.append(vlad)
    vlads = torch.stack(vlads).permute(1,0,2).reshape(len(masks),-1)
    vlads = F.normalize(vlads, dim=1)

    end_time = time.time()
    execution_time = end_time - start_time
    # return vlads
    return vlads, execution_time


def save_res_label(desc_path, vlad):
        f = h5py.File(desc_path, "r")
        keys = list(f.keys())
        res_list =[]
        labels=[]
        i=0
        for i in tqdm(range(len(keys))):
                mask_list=[]
                dino_desc = torch.from_numpy(f[keys[i]]['ift_dino'][()]).to('cuda')
                # dino_desc = torch.nn.functional.interpolate(dino_desc, [cfg['desired_height'],cfg['desired_width']], mode="bilinear", align_corners=True)
                dino_desc = torch.from_numpy(np.reshape(f[keys[i]]['ift_dino'][()],(1,1536,f[keys[i]]['ift_dino'][()].shape[2]*f[keys[i]]['ift_dino'][()].shape[3]))).to('cuda')
                # dino_desc=torch.reshape(dino_desc,(1,1536,cfg['desired_height']*cfg['desired_width'])).to('cuda')
                dino_desc_norm = torch.nn.functional.normalize(dino_desc, dim=1)
                # for j in range(len(segRange[i])):
                #         mask = torch.from_numpy(segMask[i][j]).to('cpu')
                #         # mask = torch.from_numpy(G.nodes[segRange[i][j]]['segmentation']).to('cuda')
                #         mask = torch.nn.functional.interpolate(mask.float().unsqueeze(0).unsqueeze(0), [cfg['desired_height']//14,cfg['desired_width']//14], mode = 'nearest').squeeze().bool()
                #         mask_list.append(mask.flatten())
                # reg_feat_norm = dino_desc_norm[:,:,mask]
                reg_feat_per = dino_desc_norm.permute(0,2,1)
                res,label = vlad.generate(reg_feat_per.cpu().squeeze(),save=True)
                # gd_np = gd.numpy()
                # imFts= np.vstack([imFts,gd_np])
                res_list.append(res)
                # labels.append(label)
        return res_list

from scipy.spatial import Delaunay, ConvexHull
def getNbrsDelaunay(tri,v):
    indptr, indices = tri.vertex_neighbor_vertices
    v_nbrs = indices[indptr[v]:indptr[v+1]]
    v_nbrs = [[v,u] for u in v_nbrs]
    return v_nbrs

def nbrAGG(segRange, segFt,mask_file, ims):
    segFt_agg=[]
    with h5py.File(mask_file, "r") as f:
        for i in tqdm(range(len((ims)))):
            key = ims[i]
            masks = [f[key+f'/masks/{k}/'] for k in natsorted(f[key+'/masks/'].keys())]
            masks_seg = [mask['segmentation'] for mask in masks]
            # mask_cords = np.array([mask['point_coords'][()][0] for mask in masks])
            mask_cords = np.array([np.array(np.nonzero(mask['segmentation'][()])).mean(1)[::-1] for mask in masks])
            # print(len(mask_cords))
            if len(mask_cords) > 3:
                tri = Delaunay(mask_cords)
                nbrs, nbrsLists = [], []
                rft_da_nbrs = []
                for v in range(len(mask_cords)):
                    # print(len(mask_cords))
                    nbrsList = getNbrsDelaunay(tri, v)
                    nbrsLists.append(nbrsList)
                    segFt_agg.append(segFt[segRange[i][np.unique([[v,v]]+nbrsList)]].mean(0))
                

    return segFt_agg

def nbrMasksAGGFast(mask_file, ims, order =1):
    
    # segMaskIm=[]
    adj_mat_im=[]
    with h5py.File(mask_file, "r") as f:
        for i in tqdm(range(len((ims)))):
            key = ims[i]
            masks = [f[key+f'/masks/{k}/'] for k in natsorted(f[key+'/masks/'].keys())]
            # masks_seg = [mask['segmentation'] for mask in masks]
            # mask_cords = np.array([mask['point_coords'][()][0] for mask in masks])
            mask_cords = np.array([np.array(np.nonzero(mask['segmentation'][()])).mean(1)[::-1] for mask in masks])
            # print(len(mask_cords))
            adj_mat = torch.zeros((len(mask_cords),len(mask_cords)))
            if len(mask_cords) > 3:
                tri = Delaunay(mask_cords)
                nbrsLists=[]
                for v in range(len(mask_cords)):
                  
                    nbrsList = getNbrsDelaunay(tri, v)
                    nbrsList = np.unique([[v,v]]+nbrsList)
                    nbrsLists.append(nbrsList)
                    adj_mat[v][nbrsList]=1
                if order ==1:
                        adj_mat_im.append(adj_mat.bool())
                elif order==2:
                        adj_mat_im.append((adj_mat@adj_mat).bool())
                elif order==3:
                    adj_mat_im.append((adj_mat@adj_mat@adj_mat).bool())
                elif order==4:
                    adj_mat_im.append((adj_mat@adj_mat@adj_mat@adj_mat).bool())
                elif order==5:
                    adj_mat_im.append((adj_mat@adj_mat@adj_mat@adj_mat@adj_mat).bool())
            else:
                nbr_list = [0,1]
                for v in range(len(mask_cords)):
                    adj_mat[v][nbr_list]=1
                adj_mat_im.append(adj_mat.bool())    
    return adj_mat_im

def nbrMasksAGGFastSingle(masks_seg, order=1):
    # img_key is img_name
    # masks = [mask_in[img_key + f'/masks/{k}/'] for k in natsorted(mask_in[img_key + '/masks/'].keys())]
    # mask_cords = np.array([np.array(np.nonzero(mask['segmentation'][()])).mean(1)[::-1] for mask in masks])
    # masks = [mask_in[img_key + f'/masks/{k}/'] for k in natsorted(mask_in[img_key + '/masks/'].keys())]
    mask_cords = np.array([np.array(np.nonzero(mask_seg)).mean(1)[::-1] for mask_seg in masks_seg])
    adj_mat = torch.zeros((len(mask_cords), len(mask_cords)))
    
    if len(mask_cords) > 3:
        tri = Delaunay(mask_cords)
        for v in range(len(mask_cords)):
            nbrsList = getNbrsDelaunay(tri, v)
            nbrsList = np.unique([[v, v]] + nbrsList)
            adj_mat[v][nbrsList] = 1

        # if order ==1:
        #     adj_mat.append(adj_mat.bool())
        # elif order==2:
        #     adj_mat.append((adj_mat@adj_mat).bool())
        # elif order==3:
        #     adj_mat.append((adj_mat@adj_mat@adj_mat).bool())
        # elif order==4:
        #     adj_mat.append((adj_mat@adj_mat@adj_mat@adj_mat).bool())
        # elif order==5:
        #     adj_mat.append((adj_mat@adj_mat@adj_mat@adj_mat@adj_mat).bool())

        adj_mat_power = adj_mat.clone()  
        for _ in range(order - 1):  # Multiply original matrix order-1 times
            adj_mat_power = adj_mat_power @ adj_mat
        adj_mat = adj_mat_power.bool()

    else:
        # nbr_list = [0,1] #Important-2: In full function this is the code. Just trying out better version in single code.
        nbr_list = [0, 1] if len(mask_cords) > 1 else [0]  # Adjusting for cases with less than 2 masks
        for v in range(len(mask_cords)):
            adj_mat[v][nbr_list] = 1
        adj_mat = adj_mat.bool()

    return adj_mat


def nbrMasksAGG(segRange, segMask,mask_file, ims, order =1):
    
    segMaskIm=[]
    with h5py.File(mask_file, "r") as f:
        for i in tqdm(range(len((ims)))):
            segMask_agg=[]
            key = ims[i]
            masks = [f[key+f'/masks/{k}/'] for k in natsorted(f[key+'/masks/'].keys())]
            masks_seg = [mask['segmentation'] for mask in masks]
            # mask_cords = np.array([mask['point_coords'][()][0] for mask in masks])
            mask_cords = np.array([np.array(np.nonzero(mask['segmentation'][()])).mean(1)[::-1] for mask in masks])
            # print(len(mask_cords))
            # print("shape: ", masks_seg[0].shape)
            if len(mask_cords) > 3:
                tri = Delaunay(mask_cords)
                nbrs, nbrsLists,nbrsListsOD2=[],  [], []
                rft_da_nbrs = []
                for v in range(len(mask_cords)):
                    segMask_local=np.zeros_like(masks_seg[0]).astype('bool')
                    # print(len(mask_cords))
                    nbrsListsod2=[]
                    nbrsList = getNbrsDelaunay(tri, v)
                    nbrsList = np.unique([[v,v]]+nbrsList)
                    nbrsLists.append(nbrsList)
                    # segMask_local+=np.array(segMask2[i])[np.unique([[v,v]]+nbrsList).tolist()].sum(0).astype('bool')
                    if order ==1:
                        segMask_agg.append(np.array(segMask[i])[nbrsList.tolist()].sum(0).astype('bool'))
                        # segMask_agg.append(segMask_local)
                if order ==2:
                
                    for u in range(len(nbrsLists)):
                        segMask_local=np.zeros_like(masks_seg[0]).astype('bool')
                        list_2od=[]
                        list_2od = np.array(list_2od)
                        for w in range(len(nbrsLists[u])):

                            # nbrsListod2 = getNbrsDelaunay(tri,nbrsList[u][1])
                            #  if v ==0:
                            #       print (np.unique([[u,u]]+nbrsListod2))
                            # nbrsListsod2.append(nbrsListod2)
                            list_2od = np.concatenate((list_2od,nbrsLists[nbrsLists[u][w]])).astype('int32')
                            list_2od = np.unique(list_2od)
                            # segMask_local+=np.array(segMask[i])[nbrsLists[nbrsLists[u][w]].tolist()].sum(0).astype('bool')
                        segMask_local+=np.array(segMask[i])[list_2od.tolist()].sum(0).astype('bool')
                        segMask_agg.append(segMask_local)
                        # nbrsListsOD2.append(nbrsListod2)
                if order == 3:
                    for u in range(len(nbrsLists)):
                        segMask_local=np.zeros_like(masks_seg[0]).astype('bool')
                        list_3od=[]
                        list_3od = np.array(list_3od)
                        for w in range(len(nbrsLists[u])):
                            for x in range(len(nbrsLists[nbrsLists[u][w]])):
                                list_3od = np.concatenate((list_3od,nbrsLists[nbrsLists[nbrsLists[u][w]][x]])).astype('int32')
                                list_3od = np.unique(list_3od)
                                # segMask_local+=np.array(segMask2[i])[nbrsLists[nbrsLists[nbrsLists[u][w]][x]].tolist()].sum(0).astype('bool')
                        segMask_local+=np.array(segMask[i])[list_3od.tolist()].sum(0).astype('bool')
                        segMask_agg.append(segMask_local)
                        

                segMaskIm.append(segMask_agg)
                      
                # segMaskIm2.append(segMask_agg2)
            else:
                for v in range(len(mask_cords)):
                    segMask_agg.append(segMask[i][v])
                segMaskIm.append(segMask_agg)    
    return segMaskIm

def apply_pca_transform_from_pkl(data_tensor, pca_model_path):
    """
    Loads a PCA model from a pickle file and applies the transform to the given data tensor.

    Parameters:
    - data_tensor: A PyTorch tensor containing the data to be transformed.
    - pca_model_path: Path to the pickle file containing the fitted PCA model.

    Returns:
    - A PyTorch tensor containing the transformed data.
    """
    # Ensure data is on CPU and converted to a NumPy array for PCA transformation
    data_np = data_tensor.cpu().numpy()

    # Load the fitted PCA model from disk
    with open(pca_model_path, "rb") as file:
        pca = pickle.load(file)

    # Apply the PCA transform to the data
    transformed_data_np = pca.transform(data_np)

    # Convert the transformed data back to a PyTorch tensor
    transformed_data_tensor = torch.from_numpy(transformed_data_np)

    return transformed_data_tensor

def apply_pca_transform_from_pkl_numpy(data_np, pca_model_path):
    """
    Loads a PCA model from a pickle file and applies the transform to the given data tensor.

    Parameters:
    - data_: numpy
    - pca_model_path: Path to the pickle file containing the fitted PCA model.

    Returns:
    - A PyTorch tensor containing the transformed data.
    """

    # Load the fitted PCA model from disk
    with open(pca_model_path, "rb") as file:
        pca = pickle.load(file)

    # Apply the PCA transform to the data
    transformed_data_np = pca.transform(data_np)

    # Convert the transformed data back to a PyTorch tensor
    # transformed_data_tensor = torch.from_numpy(transformed_data_np)

    return transformed_data_np

def save_maxseg_results(gt, predicted_global, predicted_local, save_path):
    with h5py.File(save_path, 'w') as hdf:
        # Store ground truth data
        dt = h5py.special_dtype(vlen=np.dtype('int32'))
        gt_dataset = hdf.create_dataset("gt", (len(gt),), dtype=dt)
        for i, gt_entry in enumerate(gt):
            gt_dataset[i] = gt_entry

        # prepare data for HDF5 storage
        def prepare_data_for_hdf5(predictions):
            prepared_data = {}
            for prediction in predictions:
                for key, value in prediction.items():
                    # NOTE: This currently only works for lists of arrays, and arrays but not other types. Handle other types if needed in future.
                    if isinstance(value, list):  # Check if value is a list of arrays, for key like 'img_id_to_seg_id'
                        value = [item.tolist() if isinstance(item, np.ndarray) else item for item in value]
                    elif isinstance(value, np.ndarray): # for rest of keys like 'seg_id_q'
                        value = value.tolist()  # Convert array to list
                    value = json.dumps(value)  # Serialize the list to a JSON string

                    if key not in prepared_data:
                        prepared_data[key] = []
                    prepared_data[key].append(value)
            return prepared_data

        prepared_global = prepare_data_for_hdf5(predicted_global)
        prepared_local = prepare_data_for_hdf5(predicted_local)

        # Create groups for predicted global and local results
        global_group = hdf.create_group("predicted_global")
        local_group = hdf.create_group("predicted_local")

        # Store each key's data as a single dataset within the group
        for key, values in prepared_global.items():#TODO: values is a list of variable length arrays, so bug currently. 
            global_group.create_dataset(key, data=np.array(values, dtype=h5py.special_dtype(vlen=str)))

        for key, values in prepared_local.items():
            local_group.create_dataset(key, data=np.array(values, dtype=h5py.special_dtype(vlen=str)))
            
def get_matches_save(matches,gt,sims,segRangeQuery,imIndsRef,n=1,method="max_sim"):
    preds=[]
    match_info = [] # At query_img_id index: [[seg_id_q, ref_img_id, seg_id_r]]
    for i in range(len(gt)):
        if method == "max_sim":
            match = np.flip(np.argsort(sims[segRangeQuery[i]])[-50:])
            # pred_match.append(match)
            match_patch = matches[segRangeQuery[i]][match]
            pred = imIndsRef[match_patch]
            pred_top_k = first_k_unique_indices(pred,n)
            preds.append(pred_top_k)

            raise NotImplementedError("match_info not implemented for max_sim. Please only use max_seg for now as save/load to h5 files is not yet implemented for max_sim or max_seg_sim.")
            # dict_info = {'seg_id_q': match[:n], 'img_id_r': pred[:n], 'seg_id_r': match_patch[:n]}
            # match_info.append(dict_info)

        elif method == "max_seg":
            match_patch = matches[segRangeQuery[i]]
            segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
            pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-n:])]
            # sim_score_t = sim_img.T[i][pred]
            img_id_to_seg_id = [np.where(imIndsRef[match_patch]==p)[0] for p in pred] # or in other words - `pred_to_match_patch`: reference image indices sorted by max seg matches and corresponding to `match_patch` indices. In each of image IDs, which query segment indices are there?
            dict_info = {'seg_id_q': segRangeQuery[i], 'img_id_r': pred, 'seg_id_r': match_patch, 'img_id_to_seg_id': img_id_to_seg_id}
            match_info.append(dict_info)
            
            # preds.append(pred[np.flip(np.argsort(sim_score_t))])
            preds.append(pred)
        elif method =="max_seg_sim":
            match_patch = matches[segRangeQuery[i]]
            segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
            pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-6:])]
            sims_patch = sims[segRangeQuery[i]]
            sim_temp=[]
            for j in range(len(pred)):
                try:
                    sim_temp.append(np.max(sims_patch[np.where(imIndsRef[match_patch]==pred[j])[0]]))
                except:
                     print("index: ", i)
                     print("pred: ", pred[j])
                     print("imInds: ", imIndsRef[match_patch])
            pred = pred[np.flip(np.argsort(sim_temp))][:n]
            preds.append(pred)
            raise NotImplementedError("match_info not implemented for max_seg_sim. Please only use max_seg for now as save/load to h5 files is not yet implemented for max_sim or max_seg_sim.")
    return preds, match_info

def create_triplets(gt,matches_max_seg):
    triplets=[]
    for i in range(len(gt)):
        positive = -1
        negative=-1
        if len(gt[i])>0:
            for j in range(len(matches_max_seg[i]['img_id_r'])):
                if matches_max_seg[i]['img_id_r'][j] in gt[i]:
                    positive = matches_max_seg[i]['img_id_r'][j]
                    break
            for k in range(len(matches_max_seg[i]['img_id_r'])):
                if matches_max_seg[i]['img_id_r'][k] not in gt[i]:
                    negative = matches_max_seg[i]['img_id_r'][k]
                    break
        
        triplet = {'anchor': i, 'positive': positive, 'negative':negative}
        triplets.append(triplet)
    return triplets

def calc_margins(triplets,match_info_max_seg,segFt_r,segFt_q,gt):
    # TODO fix for cases where there is no positive/negative
    sim_pos_all =[]
    sim_neg_all=[]
    margins =[]
    margins_seg=[]
    margins_sum=[]
    for i in range(len(gt)):
        # i = triplets[j]['anchor']
        try:    
            if len(gt[i])>0:
                idx_pos = np.where(match_info_max_seg[i]['img_id_r']==triplets[i]['positive'])[0][0]
                idx_neg = np.where(match_info_max_seg[i]['img_id_r']==triplets[i]['negative'])[0][0]
                seg_idx_pos = match_info_max_seg[i]['img_id_to_seg_id'][idx_pos]
                seg_idx_neg = match_info_max_seg[i]['img_id_to_seg_id'][idx_neg]
                desc_idx_r_pos = match_info_max_seg[i]['seg_id_r'][seg_idx_pos]
                desc_idx_r_neg = match_info_max_seg[i]['seg_id_r'][seg_idx_neg]
                desc_idx_q_pos = match_info_max_seg[i]['seg_id_q'][seg_idx_pos]
                desc_idx_q_neg = match_info_max_seg[i]['seg_id_q'][seg_idx_neg]
                segFt_r_pos = segFt_r[desc_idx_r_pos]
                segFt_q_pos = segFt_q[desc_idx_q_pos]
                segFt_r_neg = segFt_r[desc_idx_r_neg]
                segFt_q_neg = segFt_q[desc_idx_q_neg]
                sim_pos = np.diagonal(segFt_r_pos @ segFt_q_pos.T).mean()
                sim_neg = np.diagonal(segFt_r_neg @ segFt_q_neg.T).mean()
                sim_pos_sum = np.diagonal(segFt_r_pos @ segFt_q_pos.T).sum()
                sim_neg_sum = np.diagonal(segFt_r_neg @ segFt_q_neg.T).sum()
                sim_pos_all.append(sim_pos)
                sim_neg_all.append(sim_neg)
                margins.append(sim_pos-sim_neg)
                margins_sum.append(sim_pos_sum-sim_neg_sum)
                margins_seg.append(len(seg_idx_pos)-len(seg_idx_neg))
                
        except: 
             print("error at "+str(i))
             print("j: "+str(j))
    return margins,margins_sum,margins_seg,sim_pos_all,sim_neg_all

def calc_margins_global(triplets,match_info,imFt_r,imFt_q,gt):
    margins =[]
    margins_seg=[]
    margins_sum=[]

    for i in range(len(gt)):
        try:
            if len(gt[i])>0:
                idx_pos = triplets[i]['positive']
                idx_neg = triplets[i]['negative']
                imFt_r_pos = imFt_r[idx_pos]
                imFt_r_neg = imFt_r[idx_neg]
                imFt_query = imFt_q[i]
                sim_pos = imFt_r_pos @ imFt_query.T
                sim_neg = imFt_r_neg @ imFt_query.T
                margins.append(sim_pos - sim_neg)
        except :
                print("error at "+str(i))
                print("j: "+str(j))
    return margins

def segAreaCovered(segMask1,segMask2):
    area=[]
    for i in tqdm(range (len(segMask1))):
        for j in range (len(segMask1[i])):
            mask = segMask1[i][j]
            area_cov = np.where(mask==True)[0].shape[0]
            area_cov_per = area_cov/(mask.shape[0]*mask.shape[1])
            area.append(area_cov_per)
    for i in tqdm(range (len(segMask2))):
        for j in range (len(segMask2[i])):
            mask = segMask2[i][j]
            area_cov = np.where(mask==True)[0].shape[0]
            area_cov_per = area_cov/(mask.shape[0]*mask.shape[1])
            area.append(area_cov_per)
    return area

def get_utm(paths):
    coords =[]
    for path in paths:
        
        gps_coords = float(path.split('@')[1]),float(path.split('@')[2])
        coords.append(gps_coords)
    return coords


def get_positives(utmDb,utmQ,posDistThr,retDists=False):
    # positives for evaluation are those within trivial threshold range
    # fit NN to find them, search by radius
    
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(utmDb)

    print("Using Localization Radius: ", posDistThr)
    distances, positives = knn.radius_neighbors(utmQ, radius=posDistThr)

    if retDists:
        return positives, distances
    else:
        return positives



def normalizeFeat(rfts):
    rfts = np.array(rfts).reshape([len(rfts),-1])
    rfts /= np.linalg.norm(rfts,axis=1)[:,None]
    return rfts
