"""
Upscale predicted nifti masks to 200x200x72 for classification training.
If want to visualise with numpy file, go to !!!code submissions\\training\\segmentation\\convert_nifti_to_numpy.py
"""

import numpy as np
from scipy.ndimage import zoom
import nibabel as nib
import os
from pathlib import Path
import sys

def upscale_nifti_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_dir)
    nifti_files = sorted(input_path.glob("*.nii.gz"))

    if not nifti_files:
        print(f"No .nii.gz files found in {input_dir}")
        return

    print(f"Found {len(nifti_files)} NIfTI files to process")

    processed_count = 0
    error_count = 0

    # we maintain higher-res cuz we wanna learn all the info ye 😎
    target_shape = (200, 200, 72)

    for nifti_file in nifti_files:
        try:
            print(f"Processing {nifti_file.name}...")
            img = nib.load(nifti_file)
            data = img.get_fdata()

            zoom_factors = [target_shape[i] / data.shape[i] for i in range(3)]

            """
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html
            Interpolation order:
            - 0: nearest neighbor (best for labels/segmentation masks)
            - 1: bilinear
            - 3: bicubic (best for continuous data)
            """
            data_upscaled = zoom(data, zoom_factors, order=0) # we use nearest neighbour cuz this script for masks only, worked well on beta
            # data_upscaled = zoom(data, zoom_factors, order=3) # this uh for the og image

            nib.save(nib.Nifti1Image(data_upscaled, affine=img.affine), os.path.join(output_dir, nifti_file.name))
            print(f"Saved: {nifti_file.name} ({data.shape} -> {data_upscaled.shape})")

            processed_count += 1

        except Exception as e:
            print(f"Error processing {nifti_file.name}: {e}")
            error_count += 1

    print(f"\nDone. Processed: {processed_count}, Errors: {error_count}")

if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) != 2:
        print("Usage: python upscale_predicted_nifti_masks.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = args[0]
    output_dir = args[1]

    if not os.path.isdir(input_dir):
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)

    upscale_nifti_files(input_dir, output_dir)