This is the code release for our paper "Revisit Anything: Visual Place Recognition via Image Segment Retrieval (ECCV 2024)".


# Dataset
## Dataset Download

- For datasets common with AnyLoc (i.e. Baidu, VPAir, pitts, 17places), you can download dataset from [this](https://github.com/AnyLoc/AnyLoc/issues/34#issuecomment-2162492086) for now.
- For SF-XL, MSLS and AmsterTime, find instructions [here](https://saishubodh.notion.site/SF-XL-and-MSLS-download-dataset-10e874ed2adf80e98e7dd32911152562?pvs=4). 



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

Download (preprocessed) data for the steps below [here](https://universityofadelaide.box.com/s/199q2lpvy3psm5qgfagvh25r9c51ey6b) and place them at appropriate location:
- Place `models` folder from `sam_dino_models.zip` inside `workdir_data` folder


# Running segVLAD pipeline 

You can get the segVLAD pipeline up and running in few easy and quick steps as follows.

First, see which experiment/dataset you want to run in `place_rec_global_config.py`. The default config we use in paper is `exp0_global_SegLoc_VLAD_PCA_o3`, i.e. nbr aggregation for order 3 with PCA. You can try out other orders with/without PCA if you'd like.

Then run the following scripts sequentially with one or both of these arguments. One example is: `python place_rec_main.py --dataset 17places --experiment exp0_global_SegLoc_VLAD_PCA_o3 --vocab-vlad domain`.

For full pipeline, you need to run first 3 scripts below for pre-processing (1. DINO/SAM extraction  2. (optional) generate VLAD cluster centers 3. Save pca model) and then the final 4th script to run the main segVLAD pipeline get the final results.


1. For DINO/SAM extraction:   
    ```
    python place_rec_SAM_DINO.py --dataset <> --method DINO/SAM
    ```    
    depending on which dataset and model you want to extract. 
2. (Optional) For generating VLAD cluster center given a dataset (or its DINO desc path, to be precise):   
    ```
    python vlad_c_centers_pt_gen.py --dataset <>
    ```
    NOTE: You don't need to run this step on your end, they already exist in `cache` folder. You can run this if you want to generate your own cluster centers say on a new dataset.  
3. PCA extraction after the above are done:
    ```
    place_rec_global_any_dataset_pca_extraction.py --dataset <> --experiment <> --vocab-vlad <domain/map>
    ```
4. Main SegVLAD pipeline after all the above are done:
    ```
    place_rec_main.py --dataset <> --experiment <> --vocab-vlad <domain/map> --save_results <True/False>
    ```
    If you want to save the descriptors (for offline recall calculation later on), you can set `save_results` to `True` and results will automatically saved as `experiment_name_date_time` inside `{workdir}/results/global/`.

Additionally:
- The above scripts extract results for `SegVLAD-PreT` or pretrained case. If you want to run `SegVLAD-FineT` or finetuned experiments, just append the above scripts with `_finetuned` at the end. 


### TODOs:
- Make step 3 optional by uploading all pca models.
- Make step 1 optional for few small datasets
- code to replicate SALAD and all baselines


### Acknowledgements
We borrow some of the code from AnyLoc. We thank authors of AnyLoc for making their code public. 
