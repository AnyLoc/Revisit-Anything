
# See README for more details on how to set the paths/expt configs in this script.

# Step 0: set the parent path of where all datasets lie
workdir_data = '/home/kartikgarg/workdir'

# Step 1: Set Dataset specific configurations
datasets = {
    "baidu": {
        "masks_h5_filename_r": "baidu_r_masks_320.h5",
        "masks_h5_filename_q": "baidu_q_masks_320.h5",
        "dino_h5_filename_r":   "baidu_r_dino_640.h5",
        "dino_h5_filename_q":   "baidu_q_dino_640.h5",
        "dinoNV_h5_filename_r": "baidu_r_dinoNV_640.h5",
        "dinoNV_h5_filename_q": "baidu_q_dinoNV_640.h5",
        "data_subpath1_r": "training_images_undistort",
        "data_subpath2_q": "query_images_undistort",
        "cfg": {'rmin': 0, 'desired_width': 640, 'desired_height': 480}, 
        "map_vlad_cluster": "baidu",
        "domain_vlad_cluster": "indoor",
    },
    "17places": {
        "masks_h5_filename_r":       "17places_r_masks_320.h5",
        "masks_h5_filename_q":       "17places_q_masks_320.h5",
        "dino_h5_filename_r":        "17places_r_dino_640.h5",
        "dino_h5_filename_q":        "17places_q_dino_640.h5",
        "dinoNV_h5_filename_r":      "17places_r_dinoNV_640.h5",
        "dinoNV_h5_filename_q":      "17places_q_dinoNV_640.h5",
        "dinoSALAD_h5_filename_r":   "17places_r_dinoSALAD_640.h5",
        "dinoSALAD_h5_filename_q":   "17places_q_dinoSALAD_640.h5",
        "data_subpath1_r": "ref",
        "data_subpath2_q": "query",
        "cfg": {'rmin': 0, 'desired_width': 640, 'desired_height': 480}, 
        "map_vlad_cluster": "17places",
        "domain_vlad_cluster": "indoor", 
    },
    "SFXL": {
        "masks_h5_filename_r":  "SFXL_r_masks_256.h5",
        "masks_h5_filename_q":  "SFXL_q_masks_256.h5",
        "dino_h5_filename_r":   "SFXL_r_dino_512.h5",
        "dino_h5_filename_q":   "SFXL_q_dino_512.h5",
        "dinoNV_h5_filename_r":      "SFXL_r_dinoNV_512.h5",
        "dinoNV_h5_filename_q":      "SFXL_q_dinoNV_512.h5",
        "dinoSALAD_h5_filename_r":   "SFXL_r_dinoSALAD_512.h5",
        "dinoSALAD_h5_filename_q":   "SFXL_q_dinoSALAD_512.h5",
        "data_subpath1_r": "database",
        "data_subpath2_q": "queries",
        "cfg": {'rmin': 0, 'desired_width': 512, 'desired_height': 512}, 
        "map_vlad_cluster": "SFXL",
        "domain_vlad_cluster": "urban", 
    },
    "InsideOut": {
        "masks_h5_filename_r":       "InsideOut_r_masks_320.h5",
        "masks_h5_filename_q":       "InsideOut_q_masks_320.h5",
        "dino_h5_filename_r":        "InsideOut_r_dino_640.h5",
        "dino_h5_filename_q":        "InsideOut_q_dino_640.h5",
        "dinoNV_h5_filename_r":      "InsideOut_r_dinoNV_640.h5",
        "dinoNV_h5_filename_q":      "InsideOut_q_dinoNV_640.h5",
        "dinoSALAD_h5_filename_r":   "InsideOut_r_dinoSALAD_640.h5",
        "dinoSALAD_h5_filename_q":   "InsideOut_q_dinoSALAD_640.h5",
        "data_subpath1_r": "ref_images",
        "data_subpath2_q": "query_images",
        "cfg": {'rmin': 0, 'desired_width': 640, 'desired_height': 480}, 
        "map_vlad_cluster": "InsideOut",
        "domain_vlad_cluster": "urban", 
    },
    "mslsSF": {
        "masks_h5_filename_r":  "mslsSF_r_masks_320.h5",
        "masks_h5_filename_q":  "mslsSF_q_masks_320.h5",
        "dino_h5_filename_r":   "mslsSF_r_dino_640.h5",
        "dino_h5_filename_q":   "mslsSF_q_dino_640.h5",
        "dinoNV_h5_filename_r":   "mslsSF_r_dinoNV_640.h5",
        "dinoNV_h5_filename_q":   "mslsSF_q_dinoNV_640.h5",
        "dinoSALAD_h5_filename_r":   "mslsSF_r_dinoSALAD_640.h5",
        "dinoSALAD_h5_filename_q":   "mslsSF_q_dinoSALAD_640.h5",
        "data_subpath1_r": "database",
        "data_subpath2_q": "query",
        "cfg": {'rmin': 0, 'desired_width': 640, 'desired_height': 480}, 
        "map_vlad_cluster": "mslsSF",
        "domain_vlad_cluster": "urban", 
    },
    "mslsCPH": {
        "masks_h5_filename_r":  "mslsCPH_r_masks_320.h5",
        "masks_h5_filename_q":  "mslsCPH_q_masks_320.h5",
        "dino_h5_filename_r":   "mslsCPH_r_dino_640.h5",
        "dino_h5_filename_q":   "mslsCPH_q_dino_640.h5",
        "dinoNV_h5_filename_r":   "mslsCPH_r_dinoNV_640.h5",
        "dinoNV_h5_filename_q":   "mslsCPH_q_dinoNV_640.h5",
        "dinoSALAD_h5_filename_r":   "mslsCPH_r_dinoSALAD_640.h5",
        "dinoSALAD_h5_filename_q":   "mslsCPH_q_dinoSALAD_640.h5",
        "data_subpath1_r": "database",
        "data_subpath2_q": "query",
        "cfg": {'rmin': 0, 'desired_width': 640, 'desired_height': 480}, 
        "map_vlad_cluster": "mslsCPH",
        "domain_vlad_cluster": "urban", 
    },
    "VPAir": {
        "masks_h5_filename_r": "VPAir_r_masks_400.h5",
        "masks_h5_filename_q": "VPAir_q_masks_400.h5",
        "dino_h5_filename_r": "VPAir_r_dino_800.h5",
        "dino_h5_filename_q": "VPAir_q_dino_800.h5",
        "dinoNV_h5_filename_r":      "VPAir_r_dinoNV_800.h5",
        "dinoNV_h5_filename_q":      "VPAir_q_dinoNV_800.h5",
        "dinoSALAD_h5_filename_r":   "VPAir_r_dinoSALAD_800.h5",
        "dinoSALAD_h5_filename_q":   "VPAir_q_dinoSALAD_800.h5",
        "data_subpath1_r": "reference_views",
        "data_subpath2_q": "queries",
        "cfg": {'rmin': 0, 'desired_width': 800, 'desired_height': 600}, 
        "map_vlad_cluster": "VPAir",
        "domain_vlad_cluster": "aerial",
    },
    "pitts": {
        "masks_h5_filename_r": "pitts30k_r_masks.h5",
        "masks_h5_filename_q": "pitts30k_q_masks.h5",
        "dino_h5_filename_r": "pitts30k_r_dino_640.h5",
        "dino_h5_filename_q": "pitts30k_q_dino_640.h5",
        "dinoNV_h5_filename_r":      "pitts_r_dinoNV_640.h5",
        "dinoNV_h5_filename_q":      "pitts_q_dinoNV_640.h5",
        "data_subpath1_r": "pitts30k/images/test/database",
        "data_subpath2_q": "pitts30k/images/test/queries",
        "data_subpath2_q_small": "pitts30k/images/test/queries_small",
        "cfg": {'rmin': 0, 'desired_width': 640, 'desired_height': 480}, 
        "map_vlad_cluster": "pitts",
        "domain_vlad_cluster": "urban",
    },
    "AmsterTime": {
        "masks_h5_filename_r": "AmsterTime_new_masks.h5",
        "masks_h5_filename_q": "AmsterTime_old_masks.h5",
        "dino_h5_filename_r": "AmsterTime_r_dino_256.h5",
        "dino_h5_filename_q": "AmsterTime_q_dino_256.h5",
        "dinoNV_h5_filename_r":      "AmsterTime_r_dinoNV_256.h5",
        "dinoNV_h5_filename_q":      "AmsterTime_q_dinoNV_256.h5",
        "data_subpath1_r": "new",
        "data_subpath2_q": "old",
        "data_subpath2_q_small": "old_small",
        "cfg": {'rmin': 0, 'desired_width': 256, 'desired_height': 256}, 
        "map_vlad_cluster": "AmsterTime",
        "domain_vlad_cluster": "urban",
    }
}

# Step 2: Set General experiment configurations
experiments = {
    # segVLAD: default configuration used in our paper: nbr agg 3, pca
    "exp0_global_SegLoc_VLAD_PCA_o3": {
        "results_pkl_suffix": "_results_exp11_global_SegLoc_VLAD_PCA_o3.pkl", 
        # "results_pkl_suffix_dinoNV": "_results_exp11_global_SegLoc_VLAD_PCA_o3_dinoNV.pkl", 
        "global_method_name": "SegLoc",   
        "minArea": 0,
        "order": 3,
        "pca": True,
        "pca_model_pkl": "_r_fitted_pca_model_order3.pkl",
        "pca_model_pkl_map": "_r_fitted_pca_model_order3_map.pkl",
        "pca_model_pkl_dinoNV": "_r_fitted_pca_model_order3_dinoNV.pkl",
        "pca_model_pkl_map_dinoNV": "_r_fitted_pca_model_order3_map_dinoNV.pkl",
        # "pca_model_pkl": "_r_fitted_pca_model_order3_FastSAM.pkl",
    },

    # replicate AnyLoc's results i.e. AnyLoc-VLAD-DINOv2 config in thier paper:
    "exp1_global_Anyloc": {
        "results_pkl_suffix": "_results_exp1_global_Anyloc_VLAD.pkl", 
        "global_method_name": "AnyLoc",  
        "minArea": 0,
    },

    # other configurations of SegVLAD you can play with
    "exp4_global_SegLoc_VLAD_o0": {
        "results_pkl_suffix": "_results_exp4_global_SegLoc_VLAD_o0.pkl", 
        "global_method_name": "SegLoc",   
        "minArea": 0,
        "order": 0,
        "pca": False,
    },

    "exp8_global_SegLoc_VLAD_PCA_o0": {
        "results_pkl_suffix": "results_exp8_global_SegLoc_VLAD_PCA_o0.pkl", 
        "global_method_name": "SegLoc",   
        "minArea": 0,
        "order": 0,
        "pca": True,
        "pca_model_pkl": "_r_fitted_pca_model_order0.pkl",

    },

    # order 1
    "exp5_global_SegLoc_VLAD_o1": {
        "results_pkl_suffix": "_results_exp5_global_SegLoc_VLAD_o1.pkl", 
        "global_method_name": "SegLoc",   
        "minArea": 0,
        "order": 1,
        "pca": False,
    },

    "exp9_global_SegLoc_VLAD_PCA_o1": {
        "results_pkl_suffix": "_results_exp9_global_SegLoc_VLAD_PCA_o1.pkl", 
        "global_method_name": "SegLoc",   
        "minArea": 0,
        "order": 1,
        "pca": True,
        "pca_model_pkl": "_r_fitted_pca_model_order1.pkl",
    },

    # order 2
    "exp6_global_SegLoc_VLAD_o2": {
        "results_pkl_suffix": "_results_exp6_global_SegLoc_VLAD_o2.pkl", 
        "global_method_name": "SegLoc",   
        "minArea": 0,
        "order": 2,
        "pca": False,
    },
    "exp10_global_SegLoc_VLAD_PCA_o2": {
        "results_pkl_suffix": "_results_exp10_global_SegLoc_VLAD_PCA_o2.pkl", 
        "global_method_name": "SegLoc",   
        "minArea": 0,
        "order": 2,
        "pca": True,
        "pca_model_pkl": "_r_fitted_pca_model_order2.pkl",
    },

    # order 3
    "exp7_global_SegLoc_VLAD_o3": {
        "results_pkl_suffix": "_results_exp7_global_SegLoc_VLAD_o3.pkl", 
        "global_method_name": "SegLoc",   
        "minArea": 0,
        "order": 3,
        "pca": False,

    },

    # order 3 + pca is our "default config" exp0

}
