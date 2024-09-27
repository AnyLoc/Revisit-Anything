
import os
import shutil
import numpy as np
from pathlib import Path
import argparse

def load_npy_data(npy_path):
	dbImages = np.load(os.path.join(npy_path, 'msls_val_dbImages.npy'))
	qImages_all = np.load(os.path.join(npy_path, 'msls_val_qImages.npy'))
	qIdx = np.load(os.path.join(npy_path, 'msls_val_qIdx.npy'))
	qImages = qImages_all[qIdx]
	return dbImages, qImages

def copy_selected_images(src_dir, dest_dir, image_names):
	os.makedirs(dest_dir, exist_ok=True)
	copied_count = 0
	for img_name in image_names:
		src_path = os.path.join(src_dir, os.path.basename(img_name))
		dest_path = os.path.join(dest_dir, os.path.basename(img_name))
		if os.path.exists(src_path):
			shutil.copy2(src_path, dest_path)
			copied_count += 1
	return copied_count

def process_city(city, dbImages, qImages, dataset_path):
	input_db_dir = os.path.join(dataset_path, f"msls{city.upper()}", "database_all")
	input_q_dir = os.path.join(dataset_path, f"msls{city.upper()}", "query_all")
	
	output_db_dir = os.path.join(dataset_path, f"msls{city.upper()}", "database")
	output_q_dir = os.path.join(dataset_path, f"msls{city.upper()}", "query")
	
	db_count = copy_selected_images(input_db_dir, output_db_dir, 
									[img for img in dbImages if city in img])
	q_count = copy_selected_images(input_q_dir, output_q_dir, 
								   [img for img in qImages if city in img])
	
	return db_count, q_count

def count_images(directory):
	return len(list(Path(directory).glob('*.jpg')))

def verify_image_counts(dataset_path):
	expected_counts = {
		'CPH': {'database': 12556, 'query': 498},
		'SF': {'database': 6315, 'query': 242}
	}
	
	print("\nExpected image counts for VPR and 'Revisit-Anything' work:")
	for city, counts in expected_counts.items():
		print(f"{city}:")
		print(f"  Database: {counts['database']} images")
		print(f"  Query: {counts['query']} images")
	
	print("\nVerifying final image counts:")
	all_correct = True
	for city in ['CPH', 'SF']:
		db_dir = os.path.join(dataset_path, f"msls{city}", "database")
		q_dir = os.path.join(dataset_path, f"msls{city}", "query")
		
		db_count = count_images(db_dir)
		q_count = count_images(q_dir)
		
		db_expected = expected_counts[city]['database']
		q_expected = expected_counts[city]['query']
		
		print(f"{city}:")
		print(f"  Database: {db_count} images (Expected: {db_expected})")
		print(f"  Query: {q_count} images (Expected: {q_expected})")
		
		if db_count != db_expected or q_count != q_expected:
			all_correct = False
	
	if all_correct:
		print("\nAll image counts match the expected values. Congratulations! You can now test the VPR system Revisit Anything.")
	else:
		print("\nWARNING: Some image counts do not match the expected values.")

def main():
	parser = argparse.ArgumentParser(description="Clean up MSLS dataset by selecting specific images based on .npy files")
	parser.add_argument("--dataset_path", help="Path to the MSLS dataset", default="/scratch/saishubodh/segments_data")
	parser.add_argument("--npy_path", help="Path to the directory containing .npy files", default="/home/saishubodh/2023/segment_vpr/SegmentMap/dataloaders/msls_npy_files/")
	"""
	npy_path should contain the following files:
	msls_val_dbImages.npy
	msls_val_qImages.npy
	msls_val_qIdx.npy

	Download using:
	base_url = "https://raw.githubusercontent.com/serizba/salad/main/datasets/msls_val"

	```python
	def ensure_files_exist(base_url, file_names, local_dir):
		for file_name in file_names:
			local_path = os.path.join(local_dir, file_name)
			if not os.path.exists(local_path):
				print(f"File {file_name} not found, downloading...")
				download_file(f"{base_url}/{file_name}", local_path)
	```
	See `dataloaders/MapillaryDatasetVal.py` for more details.
	"""

	args = parser.parse_args()

	print("This script cleans up the MSLS dataset by selecting specific images based on .npy files.")
	print("It processes the 'database_all' and 'query_all' directories, creating new 'database' and 'query' directories with selected images.")
	print(f"Dataset path: {args.dataset_path}")
	print(f"NPY files path: {args.npy_path}")

	dbImages, qImages = load_npy_data(args.npy_path)
	
	for city in ['cph', 'sf']:
		db_count, q_count = process_city(city, dbImages, qImages, args.dataset_path)
		print(f"Processed {city.upper()}:")
		print(f"  Copied {db_count} database images")
		print(f"  Copied {q_count} query images")
	
	verify_image_counts(args.dataset_path)

if __name__ == "__main__":
	main()