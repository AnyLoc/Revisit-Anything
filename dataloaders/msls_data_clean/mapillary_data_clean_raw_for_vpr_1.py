
import os
import shutil
import argparse
from pathlib import Path

# City-specific paths
CITY_PATHS = {
	'cph': {
		'database': 'msls_images_vol_4/train_val/cph/database',
		'query': 'msls_images_vol_4/train_val/cph/query'
	},
	'sf': {
		'database': 'msls_images_vol_4/train_val/sf/database',
		'query': 'msls_images_vol_3/train_val/sf/query'
	}
}

def copy_images(src_dir, dest_dir):
	src_path = Path(src_dir)
	dest_path = Path(dest_dir)
	dest_path.mkdir(parents=True, exist_ok=True)
	
	image_count = 0
	for img_file in src_path.glob('*.jpg'):
		shutil.copy2(img_file, dest_path / img_file.name)
		image_count += 1
	
	return image_count

def reorganize_msls_dataset(input_path, output_path):
	input_counts = {}
	output_counts = {}
	for city, paths in CITY_PATHS.items():
		for img_type in ['database', 'query']:
			src_dir = os.path.join(input_path, paths[img_type], 'images')
			dest_dir = os.path.join(output_path, f'msls{city.upper()}', f'{img_type}_all')
			
			input_count = count_images(src_dir)
			input_counts[f"{city}_{img_type}"] = input_count
			
			print(f"Copying {city} {img_type} images...")
			output_count = copy_images(src_dir, dest_dir)
			output_counts[f"{city}_{img_type}"] = output_count
			
			print(f"Copied {output_count} images from {src_dir} to {dest_dir}")
	
	return input_counts, output_counts


def check_image_counts(output_path):
	counts = {}
	for city in CITY_PATHS.keys():
		for img_type in ['database', 'query']:
			dir_path = os.path.join(output_path, f'msls{city.upper()}', f'{img_type}_all')
			img_count = len(list(Path(dir_path).glob('*.jpg')))
			counts[f"{city.upper()}_{img_type}"] = img_count
			# print(f"{city.upper()} {img_type}: {img_count} images")
	return counts

def compare_with_original_msls(generated_counts):
	original_counts = {
		"CPH_database": 12601,
		"CPH_query": 6595,
		"SF_database": 6315,
		"SF_query": 4525
	}
	
	print("\nComparing with original MSLS dataset:")
	print("The original MSLS dataset has:")
	print(f"CPH: {original_counts['CPH_database']} database images and {original_counts['CPH_query']} query images")
	print(f"SF: {original_counts['SF_database']} database images and {original_counts['SF_query']} query images")
	
	all_match = True
	for key, original_count in original_counts.items():
		generated_count = generated_counts.get(key, 0)
		if original_count == generated_count:
			print(f"{key}: Match - Original: {original_count}, Generated: {generated_count}")
		else:
			print(f"{key}: Mismatch - Original: {original_count}, Generated: {generated_count}")
			all_match = False
	
	if all_match:
		print("All generated counts match the original MSLS dataset.")
	else:
		print("Some counts do not match the original MSLS dataset.")

def count_images(directory):
	return len(list(Path(directory).glob('*.jpg')))

def verify_image_counts(input_counts, output_counts):
	print("\nVerifying image counts:")
	all_equal = True
	for key in input_counts:
		if input_counts[key] == output_counts[key]:
			print(f"{key}: Input {input_counts[key]} = Output {output_counts[key]}")
		else:
			print(f"{key}: Input {input_counts[key]} != Output {output_counts[key]}")
			all_equal = False
	
	if all_equal:
		print("All image counts match between input and output.")
	else:
		print("Some image counts do not match between input and output.")

def main():
	user_info = """
	I. INPUT_PATH: You will need to set the INPUT_PATH variables to the path where the original MSLS dataset is stored, it looks like:
		1. folders inside INPUT_PATH are msls_images_vol_1, msls_images_vol_2, ..., msls_images_vol_6:
			download_msls.sh    msls_images_vol_1  msls_images_vol_3  msls_images_vol_5  msls_metadata    unzip_msls.sh
			msls_checksums.md5  msls_images_vol_2  msls_images_vol_4  msls_images_vol_6  msls_patch_v1.1
		2. Inside each msls_images_vol_X folder, there are nested folders like:	
			/scratch/saishubodh/MSLS_original/msls_images_vol_4/train_val/cph/database/images
	II. OUTPUT_PATH: Set the OUTPUT_PATH variable to the path where you want to store the reorganized dataset.
	"""
	print(user_info)
	parser = argparse.ArgumentParser(description="Reorganize MSLS dataset")
	parser.add_argument("--input_path", help="Path to the input MSLS dataset", default="/scratch/saishubodh/MSLS_original")
	parser.add_argument("--output_path", help="Path to the output directory", default="/scratch/saishubodh/segments_data")
	args = parser.parse_args()

	assert os.path.exists(args.input_path), f"Input path does not exist: {args.input_path}"
	assert os.path.exists(args.output_path), f"Output path does not exist: {args.output_path}"
	print(f"IMPORTANT: This script assumes that in the original input path, the SF and CPH paths are in 3rd and 4th split. If not, please modify the CITY_PATHS dictionary.")

	print(f"Input path: {args.input_path}")
	print(f"Output path: {args.output_path}")

	print("\nChecking image counts in output directories:")
	input_counts, output_counts = reorganize_msls_dataset(args.input_path, args.output_path)
	verify_image_counts(input_counts, output_counts)

	print("\nChecking image counts in output directories:")
	generated_counts = check_image_counts(args.output_path)
	compare_with_original_msls(generated_counts)

if __name__ == "__main__":
	main()