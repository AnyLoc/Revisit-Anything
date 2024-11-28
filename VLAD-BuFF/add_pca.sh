#!/bin/bash

#############################vlad_buff
expName="dnv2_NV_AB"
resume_model="./logs/lightning_logs/${expName}/checkpoints/last.ckpt"
#python add_pca.py --ckpt_state_dict --aggregation NETVLAD --num_pcs 8192 4096 --resume_train ${resume_model} --antiburst 


#############################vlad_buff_prepool
expName="dnv2_NV_PCA192_AB"
resume_model="./logs/lightning_logs/${expName}/checkpoints/last.ckpt"
#python add_pca.py --ckpt_state_dict --aggregation NETVLAD --nv_pca 192 --num_pcs 4096 --resume_train ${resume_model} --antiburst 

#############################vlad
expName="dnv2_NV"
resume_model="./logs/lightning_logs/${expName}/checkpoints/last.ckpt"
#python add_pca.py --ckpt_state_dict --aggregation NETVLAD --num_pcs 8192 --resume_train ${resume_model} 

