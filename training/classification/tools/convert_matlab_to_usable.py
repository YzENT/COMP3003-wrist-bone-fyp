"""
Convert MATLAB files to numpy or nifti files.
Usage: python convert_matlab_to_usable.py <input_directory> <output_directory> --format=npy/niigz --res=XX YY ZZ
If --format=npy, UN001.mat -> UN001_org_matlab.npy
if --format=niigz, UN001.mat -> wbone_001_0000.nii.gz

Example: python convert_matlab_to_usable.py <input_directory> <output_directory> --format=npy --res=128 128 48
"""

import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom
import nibabel as nib
import os
from pathlib import Path
import re
import sys

def process_matlab_files(input_dir, output_dir, format, target_shape):
    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_dir)
    mat_files = sorted(input_path.glob("*.mat"))

    if not mat_files:
        print(f"No .mat files found in {input_dir}")
        return

    print(f"Found {len(mat_files)} MATLAB files to process")

    processed_count = 0
    error_count = 0

    for mat_file in mat_files:
        match = re.match(r'UN(\d+)', mat_file.stem)
        if not match:
            print(f"Skipping {mat_file.name}: doesn't match UN### pattern")
            continue

        subject_num = match.group(1)

        try:
            print(f"Processing {mat_file.name}...")
            mat_data = sio.loadmat(mat_file)

            # Extract 'T1' variable
            if 'T1' not in mat_data:
                print(f"Error: 'T1' variable not found. Available: {[k for k in mat_data.keys() if not k.startswith('__')]}")
                error_count += 1
                continue

            t1_data = mat_data['T1']

            if format == "npy":
                zoom_factors = [target_shape[i] / t1_data.shape[i] for i in range(3)]
                t1_resampled = zoom(t1_data, zoom_factors, order=1)
                out_filename = f"UN{subject_num}_org_matlab.npy"
                np.save(os.path.join(output_dir, out_filename), t1_resampled)
                print(f"Saved: {out_filename} ({t1_data.shape} -> {t1_resampled.shape})")

            elif format == "niigz":
                zoom_factors = [target_shape[i] / t1_data.shape[i] for i in range(3)]
                t1_resampled = zoom(t1_data, zoom_factors, order=1)
                out_filename = f"wbone_{subject_num}_0000.nii.gz"
                nib.save(nib.Nifti1Image(t1_resampled, affine=np.eye(4)), os.path.join(output_dir, out_filename))
                print(f"Saved: {out_filename} ({t1_data.shape} -> {t1_resampled.shape})")

            processed_count += 1

        except Exception as e:
            print(f"Error processing {mat_file.name}: {e}")
            error_count += 1

    print(f"\nDone. Processed: {processed_count}, Errors: {error_count}")

if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) < 3:
        print("Usage: python convert_matlab_to_usable.py <input_dir> <output_dir> --format=npy/niigz [--res=XX YY ZZ]")
        sys.exit(1)

    input_dir = args[0]
    output_dir = args[1]

    if not os.path.isdir(input_dir):
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)

    format = None
    for arg in args:
        if arg.startswith("--format="):
            format = arg.split("=", 1)[1]

    if format not in ("npy", "niigz"):
        print("Error: --format must be 'npy' or 'niigz'")
        sys.exit(1)

    target_shape = None
    for i, arg in enumerate(args):
        if arg.startswith("--res="):
            first = int(arg.split("=", 1)[1])
            target_shape = (first, int(args[i+1]), int(args[i+2])) # this is so stupid btw

    # for arg in args:
    #     if arg.startswith("--res="):
    #         parts = arg.split("=", 1)[1].split()
    #         print(len(parts))
    #         if len(parts) == 3:
    #             target_shape = (int(parts[0]), int(parts[1]), int(parts[2]))

    if target_shape is None:
        print("Error: --res=XX YY ZZ is required")
        sys.exit(1)

    process_matlab_files(input_dir, output_dir, format, target_shape)