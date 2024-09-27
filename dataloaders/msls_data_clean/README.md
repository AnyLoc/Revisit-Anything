
There is a lot of confusion surrounding what exactly must be used for MSLS dataset for VPR evaluation. To address this and potentially save hours of time, these 2 scripts in current folder have been created. All you need to do is download the original MSLS dataset from LINK, put it in INPUT_PATH, give that as argument for script 1 below with OUTPUT_PATH where you want to save filtered output, then again provide this OUTPUT_PATH as argument to script 2. Please go through below scripts for more details, everything has been made clear inside the scripts. For `npy` files, go through `dataloaders/MapillaryDatasetVal.py` on how to download them.
script 1: `mapillary_data_clean_raw_for_vpr_1.py`
script 2: `mapillary_data_clean_raw_for_vpr_2.py`

### Step 1 (script 1): From original full dataset to only (CPH, SF) cities: You should get output like this (if successful)

Comparing with original MSLS dataset:
The original MSLS dataset has:
CPH: 12601 database images and 6595 query images
SF: 6315 database images and 4525 query images
CPH_database: Match - Original: 12601, Generated: 12601
CPH_query: Match - Original: 6595, Generated: 6595
SF_database: Match - Original: 6315, Generated: 6315
SF_query: Match - Original: 4525, Generated: 4525
All generated counts match the original MSLS dataset.

### Step 2 (script 2): From (CPH,SF) cities to filtered versions: You should get output like this (if successful)


Expected image counts for VPR and 'Revisit-Anything' work:
CPH:
  Database: 12556 images
  Query: 498 images
SF:
  Database: 6315 images
  Query: 242 images

Verifying final image counts:
CPH:
  Database: 12556 images (Expected: 12556)
  Query: 498 images (Expected: 498)
SF:
  Database: 6315 images (Expected: 6315)
  Query: 242 images (Expected: 242)

