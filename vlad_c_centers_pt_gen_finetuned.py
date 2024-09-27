import func, func_sr, func_vpr
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from AnyLoc.custom_datasets.baidu_dataloader import Baidu_Dataset
from AnyLoc.custom_datasets.aerial_dataloader import Aerial
from AnyLoc.custom_datasets.vpair_dataloader import VPAir
from AnyLoc.datasets_vg import map_builder
from AnyLoc.datasets_vg import util


import einops as ein
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
from AnyLoc.utilities import VLAD
from sklearn.decomposition import PCA
import pickle
import faiss
import json
from importlib import reload

import random




if __name__=="__main__":
	# See README_for_global_full.md for more details on how to run this script and what it is about. (TXDX-Later)
	# Usage: python place_rec_global_any_dataset.py --dataset baidu --experiment exp1_global_AnyLoc_VLAD_and_local_FI_LFM_and_crit_num_matches

	parser = argparse.ArgumentParser(description='Global Place Recognition on Any Dataset. See place_rec_global_config.py to see how to give arguments.')
	parser.add_argument('--dataset', required=True, help='Dataset name') # baidu, pitts etc
	parser.add_argument('--vocab-vlad',required=True, choices=['domain', 'map'], help='Vocabulary choice for VLAD. Options: map, domain.')
	args = parser.parse_args()

	print(f"Vocabulary choice for VLAD (domain/map) is {args.vocab_vlad}")

	# Load dataset and experiment configurations
	dataset_config = datasets.get(args.dataset, {})
	if not dataset_config:
		raise ValueError(f"Dataset '{args.dataset}' not found in configuration.")

	print(dataset_config)
	domain_prefix = dataset_config['domain_vlad_cluster'] if args.vocab_vlad == 'domain' else dataset_config['map_vlad_cluster']
	# domain = "VPAirNVFinetuned"
	domain = domain_prefix + "NVFinetuned"

	# cfg = {'rmin':0, 'desired_width':640, 'desired_height':480} # Note for later: Not using this cfg anywhere in local code currently. Should incorporate as part of local matching later.
	cfg = dataset_config['cfg']

	# if args.dataset == "pitts" or args.dataset.startswith("msls") or args.dataset == "tokyo247" or args.dataset == "torWIC":
	workdir = f'/scratch/saishubodh/segments_data/{args.dataset}/out'
	os.makedirs(workdir, exist_ok=True)
	workdir_data = '/scratch/saishubodh/segments_data'
	# else: 
	#     workdir = f'/ssd_scratch/saishubodh/segments_data/{args.dataset}/out'
	#     os.makedirs(workdir, exist_ok=True)
	#     workdir_data = '/ssd_scratch/saishubodh/segments_data'
	# save_path_results = f"{workdir}/results/"


	print("Check the following two: input and output")
	# dino_r_path = f"{workdir}/{dataset_config['dino_h5_filename_r']}"
	dino_r_path = f"{workdir}/{dataset_config['dinoNV_h5_filename_r']}"
	print("Input: ", dino_r_path)
	print("Output: ", domain, " -- would be in cache directory")
	desc_dimension = 768 # for segVLAD finetuned and SALAD


	f = h5py.File(dino_r_path, "r")
	keys = list(f.keys())
	db_desc=[]
	# imFts=torch.empty((0,49152))
	# imfts_batch = torch.empty((0,49152))
	# idx = np.empty((cfg['desired_height'],cfg['desired_width'],2)).astype('int32')
	# for i in range(cfg['desired_height']):
	#         for j in range(cfg['desired_width']):
	#                 idx[i,j] = np.array([np.clip(i//14,0,cfg['desired_height']//14-1) ,np.clip(j//14,0,cfg['desired_width']//14-1)])
	# i=0
	# for i in tqdm(range(len(keys))):

	sample_threshold = 2000 # only apply any sampling if the number of images is greater than this threshold
	if len(keys) > sample_threshold:
		print(f"Applying sampling for large dataset: {args.dataset}")
		# Apply sampling for large datasets
		sample_percentage = 0.3# 30% sampling
		random.seed(42)
		sampled_keys = random.sample(keys, k=int(len(keys) * sample_percentage))  
		keys_to_process = sampled_keys
	else:
		# Process all keys for smaller datasets
		print(f"Processing all images for dataset: {args.dataset}")
		keys_to_process = keys


	# for i in tqdm(range(0, len(keys), 2)):  # Step by 2 to get every alternate image
	for i in tqdm(range(len(keys_to_process))):

		# 2. Subsampling every 2nd pixel in both height and width, factor of 4
		key = keys_to_process[i]
		original_data = f[key]['ift_dino'][()]

		if len(keys) > sample_threshold:
			# Apply pixel subsampling for large datasets
			subsampled_data = original_data[:, :, ::2, ::2]
		else:
			# Use full resolution for smaller datasets
			subsampled_data = original_data

		# dino_desc = torch.from_numpy(np.reshape(f[keys[i]]['ift_dino'][()],(1,desc_dimension,-1))).to('cuda')
		dino_desc = torch.from_numpy(subsampled_data.reshape(1, desc_dimension, -1)).to('cuda')
		dino_desc_norm = torch.nn.functional.normalize(dino_desc, dim=1)
		db_desc.append(dino_desc_norm.permute(0,2,1).cpu())

		# Old code

		# mask_list=[]
		# dino_desc = torch.from_numpy(f[keys[i]]['ift_dino'][()]).to('cuda')
		# import pdb;pdb.set_trace()
		# dino_desc = torch.nn.functional.interpolate(dino_desc, [cfg['desired_height'],cfg['desired_width']], mode="bilinear", align_corners=True)
		# dino_desc = torch.from_numpy(np.reshape(f[keys[i]]['ift_dino'][()],(1,desc_dimension,f[keys[i]]['ift_dino'][()].shape[2]*f[keys[i]]['ift_dino'][()].shape[3]))).to('cuda')
		#rewrite the above line in a simpler way

		# dino_desc=torch.reshape(dino_desc,(1,1536,500*500)).to('cuda')
		# dino_desc_norm = torch.nn.functional.normalize(dino_desc, dim=1)
		# db_desc.append(dino_desc_norm.permute(0,2,1).cpu())
	db_desc=torch.cat(db_desc,dim=0)

	device = torch.device("cuda")
	# Dino_v2 properties (parameters)
	desc_layer: int = 31
	desc_facet: Literal["query", "key", "value", "token"] = "value"
	num_c: int = 32
	# Domain for use case (deployment environment)
	# domain: Literal["aerial", "indoor", "urban"] = "urban"
	# domain =dataset_config['domain_vlad_cluster'] #"habitat" #"eiffel"
	print("DOMAIN is ", domain, "PLEASE CHECK")
	cache_dir = '/home/saishubodh/2023/segment_vpr/SegmentMap/AnyLoc/demo/cache'
	ext_specifier = f"dinov2_vitg14/l{desc_layer}_{desc_facet}_c{num_c}"
	c_centers_file = os.path.join(cache_dir, "vocabulary", ext_specifier,
				domain, "c_centers.pt")
	c_centers_parent = os.path.dirname(c_centers_file)
	# create directory if it doesn't exist
	if not os.path.exists(c_centers_parent):
		os.makedirs(c_centers_parent)
		print("Created directory: ", c_centers_parent)
	vlad = VLAD(num_c, desc_dim=None, 
		cache_dir=os.path.dirname(c_centers_file))


	vlad.fit(ein.rearrange(db_desc, "n k d -> (n k) d"))

	print(f"Done fitting VLAD, check the output file: {c_centers_file}")
