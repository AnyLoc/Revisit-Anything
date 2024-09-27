# Source: MixVPR ~SALAD~ authors: https://github.com/amaralibey/MixVPR/blob/main/dataloaders/MapillaryDataset.py ; https://github.com/serizba/salad/blob/main/dataloaders/val/MapillaryDataset.py

from torch.utils.data import Dataset

import shutil
import numpy as np
from PIL import Image
import os
import requests
from natsort import natsorted, index_natsorted

# NOTE: you need to download the mapillary_sls dataset from  https://github.com/FrederikWarburg/mapillary_sls
# make sure the path where the mapillary_sls validation dataset resides on your computer is correct.
# the folder named train_val should reside in DATASET_ROOT path (that's the only folder you need from mapillary_sls)
# I hardcoded the groundtruth for image to image evaluation, otherwise it would take ages to run the groundtruth script at each epoch.

def download_file(url, local_path):
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError for bad responses
    with open(local_path, 'wb') as f:
        f.write(response.content)

def ensure_files_exist(base_url, file_names, local_dir):
    for file_name in file_names:
        local_path = os.path.join(local_dir, file_name)
        if not os.path.exists(local_path):
            print(f"File {file_name} not found, downloading...")
            download_file(f"{base_url}/{file_name}", local_path)


class MSLS(Dataset):
    def __init__(self, city_name, input_transform = None, GT_ROOT='/home/saishubodh/2023/segment_vpr/SegmentMap/dataloaders/msls_npy_files/'): #DATASET_ROOT='/scratch/saishubodh/segments_data', 
        """
        This class is used to load the MSLS dataset for finding ground truth. 
        It doesn't need to load actual data, only npy files at GT_ROOT have all the information.
        
        """
        # city_name = "sf" or "cph"
        # DATASET_ROOT = '../data/mapillary/'
        # GT_ROOT = './datasets/' # BECAREFUL, this is the ground truth that comes with GSV-Cities

        # FIRST ENSURE FILES EXIST, if not download them
        base_url = "https://raw.githubusercontent.com/serizba/salad/main/datasets/msls_val"

        # List of file names to check and download
        file_names = [
            "msls_val_dbImages.npy",
            "msls_val_pIdx.npy",
            "msls_val_qIdx.npy",
            "msls_val_qImages.npy"
        ]

        os.makedirs(GT_ROOT, exist_ok=True)
        ensure_files_exist(base_url, file_names, GT_ROOT) 

        # DATASET_ROOT is DATASET_ROOT combined with mslsSF if city_name is sf else mslsCPH
        # self.DATASET_ROOT = os.path.join(DATASET_ROOT, f"msls{city_name.upper()}")
        self.input_transform = input_transform

        # hard coded reference image names, this avoids the hassle of listing them at each epoch.
        self.dbImages = np.load(GT_ROOT+'msls_val_dbImages.npy') #18871 elements

        # hard coded index of query images
        self.qIdx = np.load(GT_ROOT+'msls_val_qIdx.npy')

        # hard coded query image names.
        self.qImages_all = np.load(GT_ROOT+'msls_val_qImages.npy')  #Has 747 images, 7 unnecessary images
        self.qImages = self.qImages_all[self.qIdx]# Has 740 images    #print(len(msls_dataset.qImages[msls_dataset.qIdx]))

        # hard coded groundtruth (correspondence between each query and its matches) 
        self.ground_truth = np.load(GT_ROOT+'msls_val_pIdx.npy', allow_pickle=True) #or self.pIdx # 740 elements
        
        # reference images then query images
        # self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx])) ## concatenate reference images then query images so that we can use only one dataloader
    
        # we need to keeo the number of references so that we can split references-queries 
        # when calculating recall@K

        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        city_datasets = self.segregate_data_by_city(self.qImages, self.dbImages, self.ground_truth)

        # NATSORTING: giving corresponding IDs between query and ref based on natsorted order 
        city_data_dict = {}
        for city_data in city_datasets:
            qImages_city, dbImages_city, ground_truth_city = city_data
            gt_city, nat_qImages_city, nat_dbImages_city = self.gt_after_natsorting_images(qImages_city, dbImages_city, ground_truth_city)
            city_name_temp = 'cph' if 'cph' in qImages_city[0] else 'sf'
            city_data_dict[city_name_temp] = (gt_city, nat_qImages_city, nat_dbImages_city)
            # print(f"Processed data for {('Copenhagen' if 'cph' in qImages_city[0] else 'San Francisco')}")
            # print("First 5 Query Images: ", nat_qImages_city[:5])
            # print("First 5 Database Images: ", nat_dbImages_city[:5])
            # print("First 5 Ground Truths: ", [gt_city[i] for i in range(5)])
            # for i in range(5): 
            #     print(f"Query Image: {nat_qImages_city[i]}")
            #     print(f"Matching Reference Images: {[nat_dbImages_city[idx] for idx in gt_city[i]]}")

        gt_sf, nat_qImages_sf, nat_dbImages_sf = city_data_dict['sf']
        gt_cph, nat_qImages_cph, nat_dbImages_cph = city_data_dict['cph']
        nat_dbImages_cph, nat_dbImages_sf, nat_qImages_cph, nat_qImages_sf = map(simplify_image_names, [nat_dbImages_cph, nat_dbImages_sf, nat_qImages_cph, nat_qImages_sf])

        # if city_name is sf then soft_positivies_per_query is gt_sf else gt_cph
        if city_name == 'sf':
            # gt_sf = np.array([np.array(sublist) for sublist in gt_sf])
            self.soft_positives_per_query = gt_sf
            self.nat_qImages = nat_qImages_sf
            self.nat_dbImages = nat_dbImages_sf
        elif city_name == 'cph':
            # gt_cph = np.array([np.array(sublist) for sublist in gt_cph])
            self.soft_positives_per_query = gt_cph
            self.nat_qImages = nat_qImages_cph
            self.nat_dbImages = nat_dbImages_cph
        else:
            raise ValueError("city_name should be 'sf' or 'cph'") 

    
    # def __getitem__(self, index):
    #     # TODO: Need to add path based on whether it is query or database. Currently that is missing.
    #     img = Image.open(self.DATASET_ROOT + self.images[index])

    #     if self.input_transform:
    #         img = self.input_transform(img)

    #     return img, index

    def __len__(self):
        return len(self.images)

    def segregate_data_by_city(self, qImages, dbImages, ground_truth):
        qImages_cph, dbImages_cph, ground_truth_cph = [], [], []
        qImages_sf, dbImages_sf, ground_truth_sf = [], [], []
        index_map_cph, index_map_sf = {}, {}
        new_index_cph, new_index_sf = 0, 0

        # filter dbImages and create mappings
        for index, image in enumerate(dbImages):
            if 'cph' in image:
                dbImages_cph.append(image)
                index_map_cph[index] = new_index_cph
                new_index_cph += 1
            elif 'sf' in image:
                dbImages_sf.append(image)
                index_map_sf[index] = new_index_sf
                new_index_sf += 1

        # filter qImages and adjust ground_truth
        for q_index, q_image in enumerate(qImages):
            if 'cph' in q_image:
                qImages_cph.append(q_image)
                ground_truth_cph.append([index_map_cph[idx] for idx in ground_truth[q_index] if idx in index_map_cph])
            elif 'sf' in q_image:
                qImages_sf.append(q_image)
                ground_truth_sf.append([index_map_sf[idx] for idx in ground_truth[q_index] if idx in index_map_sf])

        return (qImages_cph, dbImages_cph, ground_truth_cph), (qImages_sf, dbImages_sf, ground_truth_sf)

    def gt_after_natsorting_images(self, qImages, dbImages, ground_truth):
        # natsorting images and retrieving the sort index
        q_sort_order = index_natsorted(qImages)
        db_sort_order = index_natsorted(dbImages)

        # apply the sorting
        nat_qImages = np.array(qImages)[q_sort_order]
        nat_dbImages = np.array(dbImages)[db_sort_order]

        # a reverse lookup for dbImages
        query_index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(q_sort_order)}
        reverse_db_index = {old_idx: new_idx for new_idx, old_idx in enumerate(db_sort_order)}

        # ground truth to correspond to sorted dbImages
        gt = [None] * len(qImages) 
        for old_query_idx, refs in enumerate(ground_truth):
            new_query_idx = query_index_mapping[old_query_idx]
            new_refs = [reverse_db_index[old_ref_idx] for old_ref_idx in refs]
            gt[new_query_idx] = new_refs 


        return gt, nat_qImages, nat_dbImages

def copy_images(msls_dataset_qImages):
    base_source_dir = "/scratch/saishubodh/segments_data/VPR-datasets-downloader/datasets_val/mapillary_sls/msls_images_val/"

    dest_dirs = {
        'cph': "/scratch/saishubodh/segments_data/mslsCPH/queries/",
        'sf': "/scratch/saishubodh/segments_data/mslsSF/queries/"
    }

    # Simulation of your msls_dataset.qImages list
    # msls_dataset_qImages = [
    #     'train_val/cph/query/images/1RKCGBAWsZbi5dj3vR2mlw.jpg',
    #     'train_val/cph/query/images/pVdypg8zwA_g82Ia2kyLMw.jpg',
    #     'train_val/cph/query/images/1r2GmjwMWvMQ7zg7ehnYQg.jpg',
    #     'train_val/cph/query/images/ipcviasnQ0oBuoUqLGmIng.jpg',
    #     'train_val/sf/query/images/ipcviasnQ0oBuoUqLGmIng.jpg',
    #     'train_val/sf/query/images/ipcviasnQ0oBuoUqLGmIng.jpg',
    #     'train_val/cph/query/images/TIUkNk_HvwbtiHvgsoA3nw.jpg'
    # ]

    for img_path in msls_dataset_qImages:
        city = img_path.split('/')[1]  # this gets 'cph' or 'sf' from the path
        print(city)

        full_source_path = os.path.join(base_source_dir, img_path)

        destination_dir = dest_dirs[city]

        # # Make sure the destination directory exists, if not create it
        # if not os.path.exists(destination_dir):
        #     os.makedirs(destination_dir)

        destination_path = os.path.join(destination_dir, os.path.basename(img_path))

        shutil.copy(full_source_path, destination_path)
        print(f"Copied {full_source_path} to {destination_path}")
    print("Totally copied ", len(msls_dataset_qImages), " images")


def simplify_image_names(image_list):
    return [image.split('/')[-1] for image in image_list]

def clean_up_extra_images_for_mslsCPH(image_list):
    source_folder = '/scratch/saishubodh/segments_data/mslsCPH/database_full'
    destination_folder = '/scratch/saishubodh/segments_data/mslsCPH/database'
    os.makedirs(destination_folder, exist_ok=True)

    for image_name in image_list:
        source_path = os.path.join(source_folder, image_name)
        destination_path = os.path.join(destination_folder, image_name)
        shutil.copy2(source_path, destination_path)

    print(f"Copied {len(image_list)} images to {destination_folder}")

if __name__ == "__main__":
    # Base URL where the files are located on GitHub
    # Initialize the dataset
    msls_dataset = MSLS()
    qImages_all = msls_dataset.qImages #Has 747 images, 7 unnecessary images
    qImages = qImages_all[msls_dataset.qIdx] # Has 740 images    #print(len(msls_dataset.qImages[msls_dataset.qIdx]))
    ground_truth = msls_dataset.ground_truth # 740 elements
    dbImages = msls_dataset.dbImages # 18871 elements

    # copy_images(msls_dataset.qImages[msls_dataset.qIdx])

    # print("Database Images: ", dbImages[:5])  
    # print("Query Images: ",    qImages[:5])
    # print("Ground Truth: ",    ground_truth[:5])
    # ns_dbI = natsorted(msls_dataset.dbImages)
    # print("Database Images sorted: ", ns_dbI[:5])
    # ns_qI = natsorted(msls_dataset.qImages)
    # print("Query Images sorted: ", ns_qI[:5])

    # print("Length of Database Images: ", len(dbImages))
    # print("Length of Query Images: ",    len(qImages))
    # print("Length of Ground Truth: ",    len(ground_truth))


    # gt, nat_qImages, nat_dbImages = gt_after_natsorting_images(qImages, dbImages, ground_truth)
    # import pdb; pdb.set_trace()

    # SEGREGATE BY CITY
    # (qImages_cph, dbImages_cph, ground_truth_cph), (qImages_sf, dbImages_sf, ground_truth_sf) = segregate_data_by_city(qImages, dbImages, ground_truth)
    city_datasets = segregate_data_by_city(qImages, dbImages, ground_truth)

    # create an empty dict to append gt_city, nat_qImages_city, nat_dbImages_city for each city
    city_data_dict = {}
    for city_data in city_datasets:
        qImages_city, dbImages_city, ground_truth_city = city_data
        gt_city, nat_qImages_city, nat_dbImages_city = gt_after_natsorting_images(qImages_city, dbImages_city, ground_truth_city)
        city_name = 'cph' if 'cph' in qImages_city[0] else 'sf'
        city_data_dict[city_name] = (gt_city, nat_qImages_city, nat_dbImages_city)
        # print(f"Processed data for {('Copenhagen' if 'cph' in qImages_city[0] else 'San Francisco')}")
        # print("First 5 Query Images: ", nat_qImages_city[:5])
        # print("First 5 Database Images: ", nat_dbImages_city[:5])
        # print("First 5 Ground Truths: ", [gt_city[i] for i in range(5)])
        # for i in range(5): 
        #     print(f"Query Image: {nat_qImages_city[i]}")
        #     print(f"Matching Reference Images: {[nat_dbImages_city[idx] for idx in gt_city[i]]}")
        print("\n CITY OVER \n")

    gt_sf, nat_qImages_sf, nat_dbImages_sf = city_data_dict['sf']
    gt_cph, nat_qImages_cph, nat_dbImages_cph = city_data_dict['cph']
    nat_dbImages_cph, nat_dbImages_sf, nat_qImages_cph, nat_qImages_sf = map(simplify_image_names, [nat_dbImages_cph, nat_dbImages_sf, nat_qImages_cph, nat_qImages_sf])
    print("done")