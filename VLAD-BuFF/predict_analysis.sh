dataset=MSLS

path="./logs/lightning_logs/"

your_method="dnv2_NV_AB"
your_method_ckpt="wpca8192_last"

baseline="dnv2_NV"
baseline_ckpt="wpca8192_last"

python predictions.py  --dataset_name ${dataset}  --your_method_path ${path}${your_method}/${your_method_ckpt}.ckpt_${dataset}_predictions.npz --baseline_paths ${path}${baseline}/${baseline_ckpt}.ckpt_${dataset}_predictions.npz

python cluster_analysis.py  --dataset_name ${dataset} --method_our ${your_method} --baseline_name ${baseline} --your_method_path ${path}${your_method}/${your_method_ckpt}.ckpt_${dataset}_predictions.npz --baseline_path ${path}${baseline}/${baseline_ckpt}.ckpt_${dataset}_predictions.npz
