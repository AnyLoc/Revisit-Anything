############################ vlad_buff
Name=dnv2_NV_AB
wpca=8192

if [ $wpca -eq 8192 ]; then
  wpca_str="wpca8k"
elif [ $wpca -eq 4096 ]; then
  wpca_str="wpca4k"
fi

expName="${Name}_${wpca_str}"
resume_model="./logs/lightning_logs/${Name}/checkpoints/wpca${wpca}_last.ckpt"
save_dir="./logs/lightning_logs/${Name}/"
#python eval.py --aggregation NETVLAD --clusters_num 64 --resize 322 322 --wpca --num_pcs ${wpca} --antiburst --ckpt_state_dict --val_datasets MSLS pitts30k_val SPED pitts250k_test Nordland tokyo247 st_lucia amstertime baidu sfsm MSLS_Test --expName ${expName} --resume_train ${resume_model} --store_eval_output --save_dir ${save_dir} --storeSAB --no_wandb

############################ vlad_buff_prepool
nv_pca=192
wpca=4096

if [ $wpca -eq 8192 ]; then
  wpca_str="wpca8k"
elif [ $wpca -eq 4096 ]; then
  wpca_str="wpca4k"
fi

expName="dnv2_NV_PCA${nv_pca}_AB_${wpca_str}"
resume_model="./logs/lightning_logs/dnv2_NV_PCA${nv_pca}_AB/checkpoints/wpca${wpca}_last.ckpt"
save_dir="./logs/lightning_logs/dnv2_NV_PCA${nv_pca}_AB/"
#python eval.py --aggregation NETVLAD --nv_pca ${nv_pca} --clusters_num 64 --resize 322 322 --wpca --num_pcs ${wpca} --antiburst --ckpt_state_dict --val_datasets MSLS pitts30k_val SPED pitts250k_test Nordland tokyo247 st_lucia amstertime baidu sfsm MSLS_Test --expName ${expName} --resume_train ${resume_model} --store_eval_output --save_dir ${save_dir}  --no_wandb

############### no wpca
nv_pca=192
expName="dnv2_NV_PCA${nv_pca}_AB"
resume_model="./logs/lightning_logs/dnv2_NV_PCA${nv_pca}_AB/checkpoints/last.ckpt"
save_dir="./logs/lightning_logs/dnv2_NV_PCA${nv_pca}_AB/"

#python eval.py --aggregation NETVLAD --nv_pca ${nv_pca} --clusters_num 64 --resize 322 322 --antiburst --ckpt_state_dict --val_datasets MSLS pitts30k_val SPED pitts250k_test Nordland tokyo247 st_lucia amstertime baidu sfsm MSLS_Test --expName ${expName} --resume_train ${resume_model} --store_eval_output --save_dir ${save_dir} --no_wandb


############################ vlad
Name=dnv2_NV
wpca=8192

if [ $wpca -eq 8192 ]; then
  wpca_str="wpca8k"
elif [ $wpca -eq 4096 ]; then
  wpca_str="wpca4k"
fi

expName="${Name}_${wpca_str}"
resume_model="./logs/lightning_logs/${Name}/checkpoints/wpca${wpca}_last.ckpt"
save_dir="./logs/lightning_logs/${Name}/"
#python eval.py --aggregation NETVLAD --clusters_num 64 --resize 322 322 --wpca --num_pcs ${wpca} --ckpt_state_dict --val_datasets MSLS SPED pitts250k_test Nordland tokyo247 st_lucia amstertime baidu sfsm pitts30k_val MSLS_Test --expName ${expName} --resume_train ${resume_model} --store_eval_output --save_dir ${save_dir} --storeSAB --no_wandb


############################ salad
Name=dnv2_salad
expName=${Name}
resume_model="./logs/lightning_logs/${Name}/checkpoints/last.ckpt"
save_dir="./logs/lightning_logs/${Name}/"
#python eval.py --aggregation SALAD --batch_size 200 --resize 322 322 --ckpt_state_dict --val_datasets Nordland tokyo247 st_lucia amstertime baidu sfsm MSLS_Test --expName ${expName} --resume_train ${resume_model} --store_eval_output --save_dir ${save_dir} --storeSOTL --no_wandb


