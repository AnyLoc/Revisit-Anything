# Revisit Anything: Visual Place Recognition via Image Segment Retrieval (ECCV 2024)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisit-anything-visual-place-recognition-via/visual-place-recognition-on-17-places)](https://paperswithcode.com/sota/visual-place-recognition-on-17-places?p=revisit-anything-visual-place-recognition-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisit-anything-visual-place-recognition-via/visual-place-recognition-on-baidu-mall)](https://paperswithcode.com/sota/visual-place-recognition-on-baidu-mall?p=revisit-anything-visual-place-recognition-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisit-anything-visual-place-recognition-via/visual-place-recognition-on-inside-out)](https://paperswithcode.com/sota/visual-place-recognition-on-inside-out?p=revisit-anything-visual-place-recognition-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisit-anything-visual-place-recognition-via/visual-place-recognition-on-vp-air-1)](https://paperswithcode.com/sota/visual-place-recognition-on-vp-air-1?p=revisit-anything-visual-place-recognition-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisit-anything-visual-place-recognition-via/visual-place-recognition-on-amstertime)](https://paperswithcode.com/sota/visual-place-recognition-on-amstertime?p=revisit-anything-visual-place-recognition-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisit-anything-visual-place-recognition-via/visual-place-recognition-on-pittsburgh-30k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-30k?p=revisit-anything-visual-place-recognition-via)

# Dataset
## Dataset Download
- To get the pipeline running quickly with a small dataset like 17places, you can download the required data from [this box link](https://universityofadelaide.box.com/s/199q2lpvy3psm5qgfagvh25r9c51ey6b). More details on this in [Dataset preparation](https://github.com/AnyLoc/Revisit-Anything/tree/main?tab=readme-ov-file#dataset-preparation) section.
- For datasets common with AnyLoc (i.e. Baidu, VPAir, pitts, 17places), you can download dataset from [this](https://github.com/AnyLoc/AnyLoc/issues/34#issuecomment-2162492086) for now.
- For SF-XL, MSLS and AmsterTime, find instructions [here](https://saishubodh.notion.site/SF-XL-and-MSLS-download-dataset-10e874ed2adf80e98e7dd32911152562?pvs=4). 

To get going quickly, it is suggested you start with a small dataset like 17places.


## Dataset preparation
Say your main directory where you would be placing all your datasets is `workdir_data`. You may need to rename each of dataset folder/subfolder names as per the dataset name in `place_rec_global_config.py`. It must look like:   


```
workdir_data/
├── baidu 
│   ├── training_images_undistort
│   ├── query_images_undistort
|   ├── out 
├── 17places 
│   ├── ref
│   ├── query 
|   ├── out 
|
and so on
```

Notes:
1. `out` folder is where all the pre-processed data will be saved and will be created by following scripts. You just need to ensure other 2 are present beforehand, which stand for reference database and query images.


Then, you can run the following commands to get the SegVLAD results for the datasets you want. (Also, you can replicate AnyLoc's results: See the config file.)

Download (preprocessed) data for the steps below [here](https://universityofadelaide.box.com/s/199q2lpvy3psm5qgfagvh25r9c51ey6b) and place them at the appropriate location:
- Place `models` folder from `sam_dino_models.zip` inside `workdir_data` folder

To get the pipeline running quickly with a small dataset like 17places, you can download the required data from [this box link](https://universityofadelaide.box.com/s/199q2lpvy3psm5qgfagvh25r9c51ey6b) and use it in following ways:
- `17places_only_dataset.zip` (61 MB): You can download this and run 3 steps (+1 optional step below).
- `17places_full.zip` (10.3 GB): If you download this, you can skip first 3 steps of below "segVLAD pipeline" and quickly run the core pipeline and replicate the results for this dataset.
- Ensure you place it appropriately as explained at the beginning of this subsection.


# Running segVLAD pipeline 
## Env Setup
We tested the environment with Python 3.8 and torch 1.11.0
```
conda env create -f segvlad.yaml
conda activate segvlad
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
cd sam
pip install -e .
```
For running the finetuned model you will need the following additional dependencies.
```
pip install pytorch_lightning==1.9.0
pip install pytorch_metric_learning
```
Note:
- Please make sure to download the packages in this order since `pytorch_lightning` can break the env. For additional info about which version of `pytorch_lightnitng` to use with torch refer to this [link](https://lightning.ai/docs/pytorch/latest/versioning.html#pytorch-support).

## Running the code
You can get the segVLAD pipeline up and running in few easy and quick steps as follows.

First, set the path where you stored datasets i.e. `workdir_data` in `place_rec_global_config.py`. Also see which experiment/dataset you want to run in `place_rec_global_config.py`. The default config we use in paper is `exp0_global_SegLoc_VLAD_PCA_o3`, i.e. nbr aggregation for order 3 with PCA. You can try out other orders with/without PCA if you'd like.

Then run the following scripts sequentially with one or both of these arguments. One example is: `python place_rec_main.py --dataset 17places --experiment exp0_global_SegLoc_VLAD_PCA_o3 --vocab-vlad domain`.

For full pipeline, you need to run first 3 scripts below for pre-processing (1. DINO/SAM extraction  2. (optional) generate VLAD cluster centers 3. Save pca model) and then the final 4th script to run the main segVLAD pipeline get the final results. (You can skip the first 3 steps in the case of 17places dataset by downloading preprocessed data as explained at the end of [Dataset preparation](https://github.com/AnyLoc/Revisit-Anything/tree/main?tab=readme-ov-file#dataset-preparation) section.)


1. For DINO/SAM extraction: (choose one of DINO or SAM)     
    ```
    python place_rec_SAM_DINO.py --dataset <> --method DINO/SAM
    ```    
    depending on which dataset and model you want to extract. 
2. (Optional) For generating VLAD cluster center given a dataset (or its DINO desc path, to be precise):   
    ```
    python vlad_c_centers_pt_gen.py --dataset <>
    ```
    NOTE: You don't need to run this step on your end, they already exist in `cache` folder. You can run this if you want to generate your own cluster centers say on a new dataset.  
3. PCA extraction after the above are done: (Choose one of domain or map, you can start out with domain first)
    ```
    python place_rec_pca.py --dataset <> --experiment <> --vocab-vlad <domain/map>
    ```
4. Main SegVLAD pipeline after all the above are done:
    ```
    python place_rec_main.py --dataset <> --experiment <> --vocab-vlad <domain/map> --save-results <True/False>
    ```
    If you want to save the descriptors (for offline recall calculation later on), you can set `save_results` to `True` and results will automatically saved as `experiment_name_date_time` inside `{workdir}/results/global/`.

Additionally:
- The above scripts extract results for `SegVLAD-PreT` or pretrained case. If you want to run `SegVLAD-FineT` or finetuned experiments, just append the above scripts with `_finetuned` at the end, with below exception:
    - For step 1, you need to run `place_rec_DINO_finetuned.py` instead of `place_rec_SAM_DINO.py` and don't need to specify `method` argument. As you had already extracted SAM before, you just need finetuned DINO extraction here, so you can run:
    ```
    python place_rec_DINO_finetuned.py --dataset <>
    ```




### Acknowledgements
We borrow some of the code from AnyLoc. We thank authors of AnyLoc for making their code public. 
