import func, func_sr, func_vpr
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from AnyLoc.custom_datasets.baidu_dataloader import Baidu_Dataset
from AnyLoc.custom_datasets.aerial_dataloader import Aerial
from AnyLoc.custom_datasets.vpair_dataloader import VPAir
from AnyLoc.custom_datasets.eiffel_dataloader import Eiffel
from dataloaders.habitat_dataloader import Habitat
from dataloaders.MapillaryDatasetVal import MSLS 
# from dataloaders.TokyoDataset import tokyo247 
from AnyLoc.datasets_vg import map_builder
from AnyLoc.datasets_vg import util

import datetime
import time
# import tracemalloc
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

current_time = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")

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

def get_recall_old(database_vectors, query_vectors, gt, analysis =False, k=5):
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


def print_memory_usage(obj, obj_name):
	# Check if the object is a PyTorch tensor and use .storage() if it is,
	# otherwise directly use sys.getsizeof() for numpy arrays and other objects.
	if hasattr(obj, 'storage'):
		size_bytes = sys.getsizeof(obj.storage())
	else:
		size_bytes = sys.getsizeof(obj)
	size_mb = size_bytes / (1024 ** 2)
	print(f"Memory usage of '{obj_name}': {size_mb:.2f} MB")

	# Additional functionality to print the data types of arrays in a list
	if isinstance(obj, list):
		data_types = []
		for arr in obj:
			if isinstance(arr, np.ndarray):
				dtype_str = str(arr.dtype)
			elif isinstance(arr, torch.Tensor):
				dtype_str = str(arr.dtype).split('.')[-1]  # Simplify the dtype string
			elif isinstance(arr, int):
				# Python's int is not fixed-size, but we can provide a more explicit hint
				dtype_str = 'Python int (unbounded)'
			else:
				dtype_str = type(arr).__name__
			data_types.append(dtype_str)
		print(f"Data types in '{obj_name}':", ', '.join(data_types))
	elif isinstance(obj, torch.Tensor):
		print(f"Data type of '{obj_name}':", str(obj.dtype).split('.')[-1])
	elif isinstance(obj, np.ndarray):
		print(f"Data type of '{obj_name}':", obj.dtype)
	else:
		print(f"Type of '{obj_name}':", type(obj).__name__)

def recall_segloc(workdir, dataset_name, experiment_config,experiment_name, segFtVLAD1, segFtVLAD2, gt, segRange2, imInds1, map_calculate, domain, save_results=True):

	# RECALL CALCULATION
	# if pca then d = 512 else d = 49152
	if experiment_config["pca"]:
		d = 1024 #512 #PCA Dimension
		print("POTENTIAL CAUSE for error: Using d in pca before index faiss as", d, "\n 1024 or 512, check properly")
	else:
		d = 49152 #VLAD Dimension
	index = faiss.IndexFlatL2(d)
	if experiment_config["pca"]:
		index.add(func.normalizeFeat(segFtVLAD1.numpy()))
		# sims, matches = index.search(func.normalizeFeat(segFtVLAD2.numpy()),100)
		sims, matches = index.search(func.normalizeFeat(segFtVLAD2.numpy()),200)
	else:
		index.add(segFtVLAD1.numpy())
		# Should you normalize here?
		# sims, matches = index.search(segFtVLAD2.numpy(), 100)
		sims, matches = index.search(segFtVLAD2.numpy(), 200)
	# matches = matches.T[0]
	if save_results:
		out_folder = f"{workdir}/results/global/"
		if not os.path.exists(out_folder):
			os.makedirs(out_folder)

		if not os.path.exists(f"{out_folder}/{experiment_name}"):
			os.makedirs(f"{out_folder}/{experiment_name}")
		
		pkl_file_results = f"{out_folder}/{experiment_name}/{dataset_name}_matches_sims_domain_{domain}__{experiment_config['results_pkl_suffix']}"
		data_to_save = {'sims': sims, 'matches': matches}
		# Saving the data to a pickle file
		with open(pkl_file_results, 'wb') as file:
			pickle.dump(data_to_save, file)
		print(f"Results saved to {pkl_file_results}")
	
	# For  now just take 50 of those 100 matches
	sims_50 = sims[:, :50]
	matches_50 = matches[:, :50]

	sims_50 =2-sims_50#.T[0]
	# matches_justfirstone = matches.T[0]
	# sims =2-sims.T[0]
	max_seg_preds = func_vpr.get_matches(matches_50,gt,sims_50,segRange2,imInds1,n=5,method="max_seg_topk_wt_borda_Im")
	# max_seg_preds = func_vpr.get_matches_old(matches,gt,sims,segRange2,imInds1,n=5,method="max_seg")
	max_seg_recalls = func_vpr.calc_recall(max_seg_preds, gt, 5)

	print("VLAD + PCA Results \n ")
	if map_calculate:
		# mAP calculation
		queries_results = func_vpr.convert_to_queries_results_for_map(max_seg_preds, gt)
		map_value = func_vpr.calculate_map(queries_results)
		print(f"Mean Average Precision (mAP): {map_value}")

	print("Max Seg Logs: ", max_seg_recalls)
	
	return max_seg_recalls

def offline_pca_and_compute_recall_VPAir_nardo(imFts1_vlad, imFts2_vlad,  gt, pca_model_path, topK_value=5):
	"""
	Applies PCA transformation on VLAD features and computes recall.

	Parameters:
	- imFts1_vlad: numpy array of VLAD features for image set 1
	- imFts2_vlad: numpy array of VLAD features for image set 2
	- func_vpr: module containing VPR (Visual Place Recognition) related functions
	- gt: Ground truth data for computing recall

	Returns:
	- recall_VLAD_pca: Recall value after applying PCA on VLAD features
	"""


	# Apply PCA transform on VLAD features for the first set of images
	imFts1_vlad = np.stack(imFts1_vlad)
	imFtsVLAD1_pca_2048 = func_vpr.apply_pca_transform_from_pkl_numpy(imFts1_vlad, pca_model_path)
	del imFts1_vlad

	# Apply PCA transform on VLAD features for the second set of images
	imFts2_vlad = np.stack(imFts2_vlad)
	imFtsVLAD2_pca_2048 = func_vpr.apply_pca_transform_from_pkl_numpy(imFts2_vlad, pca_model_path)
	del imFts2_vlad

	# Reduce dimensionality to 1024 for both sets of PCA-transformed VLAD features
	final_dimension_pca = 1024  # Target dimensionality
	imFtsVLAD1_pca_1024 = imFtsVLAD1_pca_2048[:, :final_dimension_pca]
	imFtsVLAD2_pca_1024 = imFtsVLAD2_pca_2048[:, :final_dimension_pca]

	# Normalize features and compute recall
	print("Computing PCA for image and segments...")
	recall_VLAD_pca, _ = func_vpr.get_recall(func.normalizeFeat(imFtsVLAD1_pca_1024), 
											 func.normalizeFeat(imFtsVLAD2_pca_1024), gt, k=topK_value)
	print("RESULTS for anyloc: VLAD PCA: using  offline_pca_and_compute_recall_VPAir_nardo()")
	print(recall_VLAD_pca)

	return recall_VLAD_pca

if __name__=="__main__":
	print("IMPORTANT: In segVLAD finetuned, you will need to calculate cluster centers and add them below.\
		  This is because the descriptors are different from the ones used standard segVLAD and hence,\
		  the latent space of clusters would be different.")
	save_results = True 
	map_calculate = False  #Mean average precision

	# See README_for_global_full.md for more details on how to run this script and what it is about. (TXDX-Later)
	# Usage: python place_rec_global_any_dataset.py --dataset baidu --experiment exp1_global_AnyLoc_VLAD_and_local_FI_LFM_and_crit_num_matches
	# experiment_name = "experiment_B_512dim_pca_on_vpair_borda"
	topk_value = 5#50 #10 #5

	parser = argparse.ArgumentParser(description='Global Place Recognition on Any Dataset. See place_rec_global_config.py to see how to give arguments.')
	parser.add_argument('--dataset', required=True, help='Dataset name') # baidu, pitts etc
	parser.add_argument('--experiment', required=True, help='Experiment name') # ??? what for global ? ~exp1_global_AnyLoc_VLAD_and_local_FI_LFM_and_crit_num_matches~
	parser.add_argument('--vocab-vlad',required=True, choices=['domain', 'map'], help='Vocabulary choice for VLAD. Options: map, domain.')
	args = parser.parse_args()

	print(f"Vocabulary choice for VLAD (domain/map) is {args.vocab_vlad}")

	experiment_name = f"segVLAD_finetuned_{args.experiment}_{args.dataset}_{current_time}"#"experiment_D_baidu_FastSAM_320"
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
	# domain: Literal["aerial", "indoor", "urban"] =  dataset_config['domain_vlad_cluster']
	# domain = dataset_config['domain_vlad_cluster']
	# domain = "urban"
	domain_prefix = dataset_config['domain_vlad_cluster'] if args.vocab_vlad == 'domain' else dataset_config['map_vlad_cluster']
	# domain = "VPAirNVFinetuned"
	domain = domain_prefix + "NVFinetuned"
	ext_specifier = f"dinov2_vitg14/l{desc_layer}_{desc_facet}_c{num_c}"
	# c_centers_file = os.path.join(cache_dir, "vocabulary", ext_specifier,
	#                             domain, "c_centers.pt")
	c_centers_file = os.path.join(cache_dir, "vocabulary", ext_specifier,
								domain, "c_centers.pt")
								# domain, "pitts_nv_c_centers.pt")
	print(f"IMPORTANT: domain is {domain} and cluster center file is {c_centers_file}")
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

	# dataset specific ground truth
	if args.dataset == "baidu":
		vpr_dl =Baidu_Dataset(cfg,f"{workdir_data}",'baidu') # this should probably say just 'baidu'
		gt = vpr_dl.soft_positives_per_query
	elif args.dataset == "nardo":
		vpr_dl = Aerial(cfg, f"{workdir_data}/{args.dataset}","Tartan_GNSS_test_notrotated",)
		gt = vpr_dl.soft_positives_per_query
	elif args.dataset == "mslsSF":
		vpr_dl = MSLS(city_name="sf") #Go to this class and set GT_ROOT correctly. Everything else is not needed.
		gt = vpr_dl.soft_positives_per_query
	elif args.dataset == "mslsCPH":
		vpr_dl = MSLS(city_name="cph")
		gt = vpr_dl.soft_positives_per_query
	elif args.dataset == "tokyo247":

		# database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in ims1_r]).astype(float)
		# queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in ims2_q]).astype(float)
		# val_positive_dist_threshold = 25

		# # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
		# knn = NearestNeighbors(n_jobs=-1)
		# knn.fit(database_utms)
		# soft_positives_per_query = knn.radius_neighbors(queries_utms,
		#                                                 radius=val_positive_dist_threshold,
		#                                                 return_distance=False)
		# gt = soft_positives_per_query
		pkl_results1 = f"./tokyo_gt_dump.pkl"

		# with open(pkl_results1, 'wb') as file:
		#     pickle.dump(gt, file)
		# print(f"tokyo gt dumped successfully to {pkl_results1}")
		with open(pkl_results1, 'rb') as file:
			gt = pickle.load(file)
		print("Loaded tokyo gt dump file:", pkl_results1, gt[:2])

	elif args.dataset == "pitts":
		npy_pitts_path = f"{workdir_data}/{args.dataset}/pitts30k/images/test/"
		db = np.load(f"{npy_pitts_path}database.npy")
		q = np.load(f"{npy_pitts_path}queries.npy")
		utmDb = func_vpr.get_utm(db)
		utmQ = func_vpr.get_utm(q)
		gt = func_vpr.get_positives(utmDb, utmQ,25)
	elif args.dataset == "SFXL":
		database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in ims1_r]).astype(float)
		queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in ims2_q]).astype(float)
		from sklearn.neighbors import NearestNeighbors
		positive_dist_threshold = 25
		# Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
		knn = NearestNeighbors(n_jobs=-1)
		knn.fit(database_utms)
		gt = knn.radius_neighbors(queries_utms, radius=positive_dist_threshold, return_distance=False)
	elif args.dataset == "InsideOut":
		utmDb = pickle.load(open(f"{workdir_data}/{args.dataset}/gps_db_correct.pkl", "rb"))
		utmQ = pickle.load(open(f"{workdir_data}/{args.dataset}/gps_q_new.pkl", "rb"))
		gt = func_vpr.get_positives(utmDb, utmQ, 50)
	elif args.dataset=="17places":
		# gt = []
		loc_rad = 5
		# for i in range(len(ims2_q)):
		# 	gt.append(np.arange(i-loc_rad,i+loc_rad+1))
		gt = [list(np.arange(i - loc_rad, i + loc_rad + 1)) for i in range(len(ims2_q))]

	elif args.dataset == "AmsterTime":
		gt=[]
		for i in range(len(ims1_r)):
			gt.append([i])
	elif args.dataset == "VPAir":
		vpr_dl = VPAir(cfg,f"{workdir_data}",'VPAir')
		gt = vpr_dl.soft_positives_per_query
	elif args.dataset == "eiffel":
		vpr_dl = Eiffel(cfg,f"{workdir_data}",'eiffel')#, 'test') IMPORTANT: if not working, set to 'test' and try again
		gt = vpr_dl.soft_positives_per_query
	elif args.dataset == "habitat":
		vpr_dl = Habitat(cfg,f"{workdir_data}",'habitat')
		gt = vpr_dl.soft_positives_per_query
	elif args.dataset == "habitat_swap":
		# same as before
		vpr_dl = Habitat(cfg,f"{workdir_data}",'habitat_swap')
		gt = vpr_dl.soft_positives_per_query
	else:
		raise ValueError(f"Dataset '{args.dataset}' not found in configuration.")


	if experiment_config["global_method_name"] == "SegLoc":
		# iterate over a zip of the two lists of images and must be iterating over r_id and q_id at a time:
		dh = cfg['desired_height'] // 14
		dw = cfg['desired_width'] // 14
		idx_matrix = np.empty((cfg['desired_height'], cfg['desired_width'], 2)).astype('int32')
		for i in range(cfg['desired_height']):
			for j in range(cfg['desired_width']):
				idx_matrix[i, j] = np.array([np.clip(i//14, 0, dh-1), np.clip(j//14, 0, dw-1)])
		ind_matrix = np.ravel_multi_index(idx_matrix.reshape(-1, 2).T, (dh, dw))
		ind_matrix = torch.tensor(ind_matrix, device='cuda')


		masks_r_path = f"{workdir}/{dataset_config['masks_h5_filename_r']}"
		masks_q_path = f"{workdir}/{dataset_config['masks_h5_filename_q']}"
		masks1_h5_r = h5py.File(masks_r_path, 'r')
		masks2_h5_q = h5py.File(masks_q_path, 'r')

		order = experiment_config['order']
		print("nbr agg order number: ", order)

		segRange1 = []
		segRange2 = []
		desc_dim = 768
		print("NOTE: desc_dim here in FineT is 768, different from 1536 of PreT")
		vlad_dim = 32 * desc_dim
		# segFtVLAD1 = torch.empty((0, vlad_dim)) 
		# segFtVLAD2 = torch.empty((0, vlad_dim)) 

		# For PCA
		total_segments = 0# Counter for sampled segments
		max_segments = 50000  # Max segments to sample in total

		batch_size = 100  # Number of images to process before applying PCA

		if experiment_config["pca"]:
			# offline_pca_vpair_nardo = True  #For this, check script ending with same name but _vpair_nardo_pca.py
			# if offline_pca_vpair_nardo:
			#     pca_model_path = f"/ssd_scratch/saishubodh/segments_data/nardo/out/nardo_VPAir{experiment_config['pca_model_pkl']}"
			#     # offline_pca_and_compute_recall_VPAir_nardo(imFts1_vlad, imFts2_vlad,  gt, pca_model_path)
			# else:
			# pca_model_path = f"{workdir}/{args.dataset}{experiment_config['pca_model_pkl_dinoNV']}"

			if args.vocab_vlad == 'domain': 
				pca_model_path = f"{workdir}/{args.dataset}{experiment_config['pca_model_pkl_dinoNV']}"
			elif args.vocab_vlad == 'map':
				pca_model_path = f"{workdir}/{args.dataset}{experiment_config['pca_model_pkl_map_dinoNV']}"
			else:
				raise ValueError(f"Unknown vocab-vlad value: {args.vocab_vlad}")
		else: 
			pca_model_path = None



		segFtVLAD1_list = [] 
		segFtVLAD1Pca_list = [] 
		batch_descriptors_r = [] 

		segFtVLAD2_list = [] 
		segFtVLAD2Pca_list = [] 
		batch_descriptors_q = [] 

		imInds1 = np.array([], dtype=int)
		imInds2 = np.array([], dtype=int)

		execution_times_total = []

		print("Computing SegLoc for all images in the dataset...")
		for r_id, r_img in tqdm(enumerate(ims1_r), total=len(ims1_r), desc="Processing for reference images..."):

			# print(r_id, r_img)
			# Preload all masks for the image
			masks_seg = func_vpr.preload_masks(masks1_h5_r, r_img)

			imInds1_ind, regInds1_ind, segMask1_ind = func_vpr.getIdxSingleFast(r_id,masks_seg,minArea=experiment_config['minArea'])

			imInds1 = np.concatenate((imInds1, imInds1_ind))

			if order: 
				adjMat1_ind = func_vpr.nbrMasksAGGFastSingle(masks_seg, order)
			else:
				adjMat1_ind = None

			gd = func_vpr.seg_vlad_gpu_single(ind_matrix, idx_matrix, dino1_h5_r, r_img, segMask1_ind, c_centers, cfg, desc_dim, adj_mat=adjMat1_ind)
			# gd, execution_time  = func_vpr.seg_vlad_gpu_single(ind_matrix, idx_matrix, dino1_h5_r, r_img, segMask1_ind, c_centers, cfg, desc_dim=768, adj_mat=adjMat1_ind)
			# execution_times_total.append(execution_time)

			if experiment_config["pca"]:
				batch_descriptors_r.append(gd)
				if (r_id + 1) % batch_size == 0 or (r_id + 1) == len(ims1_r): ## Once we have accumulated descriptors for 100 images or are at the last image, process the batch
					segFtVLAD1_batch = torch.cat(batch_descriptors_r, dim=0)
					# Reset batch descriptors for the next batch
					batch_descriptors_r = []

					print("Applying PCA to batch descriptors... at image ", r_id)
					segFtVLAD1Pca_batch  = func_vpr.apply_pca_transform_from_pkl(segFtVLAD1_batch, pca_model_path)
					del segFtVLAD1_batch

					segFtVLAD1Pca_list.append(segFtVLAD1Pca_batch)
					

			else:
				# segFtVLAD1 = torch.cat((segFtVLAD1, imfts_batch), dim=0) # instead of this, we will append to a list and then cat at the end
				segFtVLAD1_list.append(gd) #imfts_batch same as gd here, in the full image function, it is for 100 images at a time
				# segFtVLAD1_list.append(gd)



		# average_time = sum(execution_times_total) / len(execution_times_total)
		# print(f"Average execution time for vlad_matmuls_per_cluster: {average_time} seconds")


		# segFtVLAD1 = torch.cat(segFtVLAD1_list, dim=0)
		if experiment_config["pca"]:
			segFtVLAD1 = torch.cat(segFtVLAD1Pca_list, dim=0)
			print("Shape of segment descriptors with PCA:", segFtVLAD1.shape)
			del segFtVLAD1Pca_list
		else:
			segFtVLAD1 = torch.cat(segFtVLAD1_list, dim=0)
			print("Shape of segment descriptors without PCA:", segFtVLAD1.shape)
			del segFtVLAD1_list

		for i in range(imInds1[-1]+1):
			segRange1.append(np.where(imInds1==i)[0])




		# QUERIES
		for q_id, q_img in tqdm(enumerate(ims2_q), total=len(ims2_q), desc="Processing for query images..."):
			# print("query: ", q_id, q_img)
			# Preload all masks for the image
			masks_seg = func_vpr.preload_masks(masks2_h5_q, q_img)

			imInds2_ind, regInds2_ind, segMask2_ind = func_vpr.getIdxSingleFast(q_id,masks_seg,minArea=experiment_config['minArea'])

			imInds2 = np.concatenate((imInds2, imInds2_ind))

			if order: 
				adjMat2_ind = func_vpr.nbrMasksAGGFastSingle(masks_seg, order)
			else:
				adjMat2_ind = None

			gd = func_vpr.seg_vlad_gpu_single(ind_matrix, idx_matrix, dino2_h5_q, q_img, segMask2_ind, c_centers, cfg, desc_dim, adj_mat=adjMat2_ind)


			if experiment_config["pca"]:
				batch_descriptors_q.append(gd)
				if (q_id + 1) % batch_size == 0 or (q_id + 1) == len(ims2_q): ## Once we have accumulated descriptors for 100 images or are at the last image, process the batch
					segFtVLAD2_batch = torch.cat(batch_descriptors_q, dim=0)
					# Reset batch descriptors for the next batch
					batch_descriptors_q = []

					print("query: Applying PCA to batch descriptors... at image ", q_id)
					segFtVLAD2Pca_batch  = func_vpr.apply_pca_transform_from_pkl(segFtVLAD2_batch, pca_model_path)
					del segFtVLAD2_batch

					segFtVLAD2Pca_list.append(segFtVLAD2Pca_batch)
					

			else:
				# segFtVLAD1 = torch.cat((segFtVLAD1, imfts_batch), dim=0) # instead of this, we will append to a list and then cat at the end
				segFtVLAD2_list.append(gd) #imfts_batch same as gd here, in the full image function, it is for 100 images at a time
				# segFtVLAD1_list.append(gd)



		if experiment_config["pca"]:
			segFtVLAD2 = torch.cat(segFtVLAD2Pca_list, dim=0)
			print("Shape of q segment descriptors with PCA:", segFtVLAD2.shape)
			del segFtVLAD2Pca_list
		else:
			segFtVLAD2 = torch.cat(segFtVLAD2_list, dim=0)
			print("Shape of q segment descriptors without PCA:", segFtVLAD2.shape)
			del segFtVLAD2_list

		for i in range(imInds2[-1]+1):
			segRange2.append(np.where(imInds2==i)[0])

		if save_results:
			out_folder = f"{workdir}/results/global/"
			if not os.path.exists(out_folder):
				os.makedirs(out_folder)

			if not os.path.exists(f"{out_folder}/{experiment_name}"):
				os.makedirs(f"{out_folder}/{experiment_name}")
			
			pkl_file_results1 = f"{out_folder}/{experiment_name}/{args.dataset}_segFtVLAD1_domain_{domain}__{experiment_config['results_pkl_suffix']}"
			pkl_file_results2 = f"{out_folder}/{experiment_name}/{args.dataset}_segFtVLAD2_domain_{domain}__{experiment_config['results_pkl_suffix']}"

			# Saving segFtVLAD1
			with open(pkl_file_results1, 'wb') as file:
				pickle.dump(segFtVLAD1, file)
			print(f"segFtVLAD1 tensor saved to {pkl_file_results1}")

			# Saving segFtVLAD2
			with open(pkl_file_results2, 'wb') as file:
				pickle.dump(segFtVLAD2, file)
			print(f"segFtVLAD2 tensor saved to {pkl_file_results2}")

		# RECALL CALCULATION
		recall_segrec = recall_segloc(workdir, args.dataset, experiment_config, experiment_name, segFtVLAD1, segFtVLAD2, gt, segRange2, imInds1, map_calculate, domain, save_results)
		print("VLAD + PCA RESULTS for segloc for dataset config: ", dataset_config, " ::: experiment config ::: ", experiment_config, " using pca file : ", pca_model_path, "experiment_name: ", experiment_name)
		print(recall_segrec)



	elif experiment_config["global_method_name"] == "AnyLoc":
		print("Running AnyLoc for global place recognition...")

		# GAP
		imFts1 = func_vpr.aggFt(dino_r_path,None,None,cfg,'avg', None,upsample=True)
		imFts2 = func_vpr.aggFt(dino_q_path,None,None,cfg,'avg', None,upsample=True)

		# VLAD
		imFts1_vlad = func_vpr.aggFt(dino_r_path,None,None,cfg,'vlad',vlad,upsample=True)
		imFts2_vlad = func_vpr.aggFt(dino_q_path,None,None,cfg,'vlad',vlad,upsample=True)

		topk_value = topk_value #10 #5
		# recall,match_info = func_vpr.get_recall(func.normalizeFeat(imFts1),func.normalizeFeat(imFts2),gt, False, k=topk_value)
		# print("RESULTS for anyloc: GAP :  ")
		# print(recall)
		# recall_vlad,match_info = func_vpr.get_recall(func.normalizeFeat(imFts1_vlad),func.normalizeFeat(imFts2_vlad),gt, False, k=topk_value)
		# print("RESULTS for anyloc: VLAD: for top ", topk_value, " matches")

		# if map_calculate:
		#     # mAP calculation
		#     formatted_max_seg_preds = [item['img_id_r'].tolist() for item in match_info]
		#     queries_results = func_vpr.convert_to_queries_results_for_map(formatted_max_seg_preds, gt)
		#     map_value = func_vpr.calculate_map(queries_results)
		#     print(f"Mean Average Precision (mAP) for VLAD: {map_value} for top {topk_value} matches.")
		# print(recall_vlad)

		offline_pca_vpair_nardo = False
		if offline_pca_vpair_nardo:
			pca_model_path = '/ssd_scratch/saishubodh/segments_data/nardo/out/nardo_VPAir_r_fitted_pca_model_anyloc.pkl'
			offline_pca_and_compute_recall_VPAir_nardo(imFts1_vlad, imFts2_vlad,  gt, pca_model_path, topk_value)
		
		else:
			#PCA initialise for image and segment features
			pca_lower_dim = 1024 #512
			pca_whiten = True
			pca_im = PCA(n_components=pca_lower_dim, whiten=pca_whiten)

			# computing pca for GAP (can repeat for VLAD)
			print ("computing pca for image and segments ...")
			imFtsVLAD1_pca = pca_im.fit_transform(func.normalizeFeat(imFts1_vlad))
			imFtsVLAD2_pca = pca_im.transform(func.normalizeFeat(imFts2_vlad))

			recall_VLAD_pca, match_info = func_vpr.get_recall(func.normalizeFeat(imFtsVLAD1_pca), func.normalizeFeat(imFtsVLAD2_pca),gt, k=topk_value)

			print("VLAD + PCA Results \n ")
			if map_calculate:
				formatted_max_seg_preds = [item['img_id_r'].tolist() for item in match_info]
				# mAP calculation
				queries_results = func_vpr.convert_to_queries_results_for_map(formatted_max_seg_preds, gt)
				map_value = func_vpr.calculate_map(queries_results)
				print(f"Mean Average Precision (mAP) for VLAD + PCA: {map_value} for top {topk_value} matches.")

			print("RESULTS for anyloc: VLAD PCA:  ")
			print(recall_VLAD_pca)

	else:
		raise ValueError(f"Global Method '{experiment_config['global_method_name']}' not found in configuration.")

	print("Script fully Executed!")
