import func, func_sr, func_vpr
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from AnyLoc.custom_datasets.baidu_dataloader import Baidu_Dataset
from AnyLoc.custom_datasets.aerial_dataloader import Aerial
from AnyLoc.datasets_vg import map_builder
from AnyLoc.datasets_vg import util

import time
import sys
# import utils
# import nbr_agg


import argparse
from place_rec_global_config import datasets, experiments


import utm
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
from AnyLoc.utilities import VLAD
from sklearn.decomposition import PCA
import pickle
import faiss
import json
from importlib import reload

# matplotlib.use('TkAgg')
matplotlib.use('Agg') #Headless

reload(func_vpr)


# from sklearn.neighbors import NearestNeighbors
# from sklearn.neighbors import KDTree


def first_k_unique_indices(ranked_indices, K):
	"""
	Obtain the first K unique indices from a ranked list of N indices.

	:param ranked_indices: List[int] - List of ranked indices
	:param K: int - Number of unique indices to obtain
	:return: List[int] - List containing first K unique indices
	"""
	seen = set()
	return [x for x in ranked_indices if x not in seen and (seen.add(x) or True)][:K]

def get_matches(matches,gt,sims,segRangeQuery,imIndsRef,n=1,method="max_sim"):
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
	print(num_eval)
	if analysis:
		return recalls.tolist(), recall_per_query
	return recalls.tolist()

def unpickle(file):
	pickle_out = open(file,'rb')
	desc = pickle.load(pickle_out)
	pickle_out.close()
	return desc

def get_recall(database_vectors, query_vectors, gt, analysis =False, k=5):
	# Original PointNetVLAD code
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
		matches.append(indices)
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
		if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
			one_percent_retrieved += 1

	one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
	recall = (np.cumsum(recall)/float(num_evaluated))*100
	print(num_evaluated)
	if analysis:
		return recall, recall_per_query, matches
	return recall,matches

def aggFt(desc_path, masks, segRange, cfg,aggType, vlad = None, upsample = False, segment_global = False,segment = False):
	f = h5py.File(desc_path, "r")
	keys = list(f.keys())
	imFts=[]
	for i in tqdm(range(len(keys))):
		if aggType =="avg":
			segfeat = torch.empty([1,768,0]).to('cuda')
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
				dino_desc = torch.from_numpy(np.reshape(f[keys[i]]['ift_dino'][()],(1,768,f[keys[i]]['ift_dino'][()].shape[2]*f[keys[i]]['ift_dino'][()].shape[3]))).to('cuda')
				# if upsample:
				#     dino_desc = torch.nn.functional.interpolate(dino_desc, [cfg['desired_height'],cfg['desired_width']], mode="bilinear", align_corners=True)
				dino_desc_norm = torch.nn.functional.normalize(dino_desc, dim=1)
				dino_desc_per = dino_desc_norm.permute(0,2,1)
				gd = vlad.generate(dino_desc_per.cpu().squeeze())
				gd_np = gd.numpy()
				imFts.append(gd_np)
	return imFts


if __name__=="__main__":
	# See README_for_global_full.md for more details on how to run this script and what it is about. (TXDX-Later)
	# Usage: python place_rec_global_any_dataset.py --dataset baidu --experiment exp1_global_AnyLoc_VLAD_and_local_FI_LFM_and_crit_num_matches

	parser = argparse.ArgumentParser(description='Global Place Recognition on Any Dataset. See place_rec_global_config.py to see how to give arguments.')
	parser.add_argument('--dataset', required=True, help='Dataset name') # baidu, pitts etc
	parser.add_argument('--experiment', required=True, help='Experiment name') # ??? what for global ? ~exp1_global_AnyLoc_VLAD_and_local_FI_LFM_and_crit_num_matches~
	parser.add_argument('--vocab-vlad',required=True, choices=['domain', 'map'], help='Vocabulary choice for VLAD. Options: map, domain.')
	parser.add_argument('--debug', action='store_true', help='Not being implemented yet.')
	args = parser.parse_args()

	print(f"Vocabulary choice for VLAD (domain/map) is {args.vocab_vlad}")
	# Load dataset and experiment configurations
	dataset_config = datasets.get(args.dataset, {})
	if not dataset_config:
		raise ValueError(f"Dataset '{args.dataset}' not found in configuration.")

	experiment_config = experiments.get(args.experiment, {})
	if not experiment_config:
		raise ValueError(f"Experiment '{args.experiment}' not found in configuration.")

	print(dataset_config)
	print(experiment_config)

	# cfg = {'rmin':0, 'desired_width':640, 'desired_height':480} # Note for later: Not using this cfg anywhere in local code currently. Should incorporate as part of local matching later.
	cfg = dataset_config['cfg']

	# if args.dataset == "pitts" or args.dataset.startswith("msls") or args.dataset == "tokyo247":
	workdir = f'/scratch/saishubodh/segments_data/{args.dataset}/out'
	os.makedirs(workdir, exist_ok=True)
	workdir_data = '/scratch/saishubodh/segments_data'
	# else: 
	#     # raise not implemented error, works only for outdoor datasets because of vlad cluster center file
	#     raise NotImplementedError("This script works only for outdoor datasets because of vlad cluster center file")
	#     # workdir = f'/ssd_scratch/saishubodh/segments_data/{args.dataset}/out'
	#     # os.makedirs(workdir, exist_ok=True)
	#     # workdir_data = '/ssd_scratch/saishubodh/segments_data'
	save_path_results = f"{workdir}/results/"

	cache_dir = '/home/saishubodh/2023/segment_vpr/SegmentMap/AnyLoc/demo/cache'

	#reading vlad clusters from cache (taken from AnyLoc anyloc_vlad_generate_colab.ipynb)
	device = torch.device("cuda")
	# Dino_v2 properties (parameters)
	desc_layer: int = 31
	desc_facet: Literal["query", "key", "value", "token"] = "value"
	num_c: int = 32
	# Domain for use case (deployment environment)
	# domain: Literal["aerial", "indoor", "urban"] =  "urban" #dataset_config['domain_vlad_cluster']
	# domain = "urban"
	domain_prefix = dataset_config['domain_vlad_cluster'] if args.vocab_vlad == 'domain' else dataset_config['map_vlad_cluster']
	# domain = "VPAirNVFinetuned"
	domain = domain_prefix + "NVFinetuned"
	ext_specifier = f"dinov2_vitg14/l{desc_layer}_{desc_facet}_c{num_c}"
	# c_centers_file = os.path.join(cache_dir, "vocabulary", ext_specifier,
	#                             domain, "c_centers.pt")
	# print("NOTE: Hardcoded cluster center file: pitts_nv_c_centers.pt")
	c_centers_file = os.path.join(cache_dir, "vocabulary", ext_specifier,
								domain, "c_centers.pt")
								# domain, "pitts_nv_c_centers.pt")
	print("NOTE:  cluster center file:", c_centers_file)
	assert os.path.isfile(c_centers_file), "Cluster centers not cached!"
	c_centers = torch.load(c_centers_file)
	assert c_centers.shape[0] == num_c, "Wrong number of clusters!"

	vlad = VLAD(num_c, desc_dim=None, 
			cache_dir=os.path.dirname(c_centers_file))
	# Fit (load) the cluster centers (this'll also load the desc_dim)
	vlad.fit(None)
	# save_path_plots = f"{workdir}/plots/{args.experiment}/"
	# if not os.path.exists(save_path_plots):
	#     os.makedirs(save_path_plots)
	# print("IMPORTANT: plots being saved at ", save_path_plots, "\n WARNING: size on disk might get too big! ")
	# # else:
	# #     save_path_plots = None

	# 0.END: Set paths and config 
	#Load Descriptors
	dataPath1_r = f"{workdir_data}/{args.dataset}/{dataset_config['data_subpath1_r']}/"
	dataPath2_q = f"{workdir_data}/{args.dataset}/{dataset_config['data_subpath2_q']}/"

	dino_r_path = f"{workdir}/{dataset_config['dinoNV_h5_filename_r']}"
	dino_q_path = f"{workdir}/{dataset_config['dinoNV_h5_filename_q']}"
	dino1_h5_r = h5py.File(dino_r_path, 'r')
	dino2_h5_q = h5py.File(dino_q_path, 'r')

	ims_sidx, ims_eidx, ims_step = 0, None, 1
	ims1_r = natsorted(os.listdir(f'{dataPath1_r}'))
	ims1_r = ims1_r[ims_sidx:ims_eidx][::ims_step]
	ims2_q = natsorted(os.listdir(f'{dataPath2_q}'))
	ims2_q = ims2_q[ims_sidx:ims_eidx][::ims_step]


	# iterate over a zip of the two lists of images and must be iterating over r_id and q_id at a time:
	dh = cfg['desired_height'] // 14
	dw = cfg['desired_width'] // 14
	idx_matrix = np.empty((cfg['desired_height'], cfg['desired_width'], 2)).astype('int32')
	for i in range(cfg['desired_height']):
		for j in range(cfg['desired_width']):
			idx_matrix[i, j] = np.array([np.clip(i//14, 0, dh-1), np.clip(j//14, 0, dw-1)])
	ind_matrix = np.ravel_multi_index(idx_matrix.reshape(-1, 2).T, (dh, dw))
	ind_matrix = torch.tensor(ind_matrix, device='cuda')

	if experiment_config["global_method_name"] == "SegLoc":
		masks_r_path = f"{workdir}/{dataset_config['masks_h5_filename_r']}"
		masks_q_path = f"{workdir}/{dataset_config['masks_h5_filename_q']}"
		masks1_h5_r = h5py.File(masks_r_path, 'r')
		masks2_h5_q = h5py.File(masks_q_path, 'r')

		print("counting masks in both r and q: ")
		numSegments1_r = func_vpr.countNumMasksInDataset(ims1_r, masks1_h5_r)
		numSegments2_q = func_vpr.countNumMasksInDataset(ims2_q, masks2_h5_q)
		print("numSegments1_r: ", numSegments1_r, "numSegments2_q: ", numSegments2_q)

		# For PCA
		accumulated_segments = 0
		max_segments = 50000  # Max segments to sample in total
		global_sampling_ratio_r = min(1, max_segments / numSegments1_r)
		# global_sampling_ratio_q = min(1, max_segments / numSegments2_q)
		# max_segments = 150000  # Max segments to sample in total
		# global_sampling_ratio_r = 1 

		pca_lower_dim = 1024 #512
		pca_whiten = True
		svd_solver = "arpack"
		pca = PCA(n_components=pca_lower_dim, whiten=pca_whiten, svd_solver=svd_solver)



		order = experiment_config['order']
		print("nbr agg order number: ", order)

		segRange1 = []
		segRange2 = []
		desc_dim = 768#1536
		vlad_dim = 32 * desc_dim
		# segFtVLAD1 = torch.empty((0, vlad_dim)) 
		# segFtVLAD2 = torch.empty((0, vlad_dim)) 


		segFtVLAD1_list = [] 
		segFtVLAD2_list = [] 
		imInds1 = np.array([], dtype=int)
		imInds2 = np.array([], dtype=int)

		print("Computing SegLoc for all images in the dataset...")
		for r_id, r_img in tqdm(enumerate(ims1_r), total=len(ims1_r), desc="Processing for reference images..."):

			print(r_id, r_img)
			# Preload all masks for the image
			masks_seg = func_vpr.preload_masks(masks1_h5_r, r_img)

			imInds1_ind, regInds1_ind, segMask1_ind = func_vpr.getIdxSingleFast(r_id,masks_seg,minArea=experiment_config['minArea'])

			imInds1 = np.concatenate((imInds1, imInds1_ind))

			if order: 
				adjMat1_ind = func_vpr.nbrMasksAGGFastSingle(masks_seg, order)
			else:
				adjMat1_ind = None

			gd = func_vpr.seg_vlad_gpu_single(ind_matrix, idx_matrix, dino1_h5_r, r_img, segMask1_ind, c_centers, cfg, desc_dim=desc_dim, adj_mat=adjMat1_ind)

			gd = gd.to(dtype=torch.float32) # Convert to float32 for PCA to keep RAM in check

			current_batch_size = gd.shape[0]
			sample_size = int(current_batch_size * global_sampling_ratio_r)
			
			if experiment_config["pca"]:
				if sample_size > 0:
					sample_indices = torch.randperm(current_batch_size)[:sample_size]
					sampled_gd = gd[sample_indices]
					segFtVLAD1_list.append(sampled_gd)
					accumulated_segments += sampled_gd.shape[0]
			else:
				# segFtVLAD1 = torch.cat((segFtVLAD1, imfts_batch), dim=0) # instead of this, we will append to a list and then cat at the end
				segFtVLAD1_list.append(gd) #imfts_batch same as gd here, in the full image function, it is for 100 images at a time
				# segFtVLAD1_list.append(gd)
			
			if accumulated_segments >= max_segments:
				break



		print("Before cat")
		segFtVLAD1 = torch.cat(segFtVLAD1_list, dim=0)
		print("After cat")
		del segFtVLAD1_list
		print("After del")

		if experiment_config["pca"]:
			print("svd solver using : ", svd_solver)
			print("NOTE: This process may take some time depending on the size of the data. \n Please do not exit...")

			pca.fit(segFtVLAD1.numpy())
			if args.vocab_vlad == 'domain': 
				pca_model_path = f"{workdir}/{args.dataset}{experiment_config['pca_model_pkl_dinoNV']}"
			elif args.vocab_vlad == 'map':
				pca_model_path = f"{workdir}/{args.dataset}{experiment_config['pca_model_pkl_map_dinoNV']}"
			else:
				raise ValueError(f"Unknown vocab-vlad value: {args.vocab_vlad}")
			# pca_model_path = f"{workdir}/{args.dataset}{experiment_config['pca_model_pkl_dinoNV']}"
			with open(pca_model_path, "wb") as file:
				pickle.dump(pca, file)

			# segFtVLAD1Pca  = func_vpr.apply_pca_transform_from_pkl(segFtVLAD1, pca_model_path)
			print("DONE: PCA for reference images (50k randomly sampled segments) and saving to pickle file")
			print(dataset_config, experiment_config)


