import numpy as np
import nibabel as nib
import os
from pathlib import Path
import re
import sys

"""
- Convert .npy files to .nii.gz format for nnUNet training.
- Expects all npy files to be in 1 folder
- Naming format of UN001_org.npy, UN001_lab.npy, UN002_org.npy, etc...
- Creates two seperate folders (org/lab)
-> org stands for original image
-> lab stands for labels
- Put respective files into nnUnet directory
"""

def convert_npy_to_niigz(input_dir, output_dir):
    org_dir = os.path.join(output_dir, "org")
    lab_dir = os.path.join(output_dir, "lab")
    os.makedirs(org_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    
    input_path = Path(input_dir)
    npy_files = list(input_path.glob("*.npy"))
    
    # Ignore _template files
    npy_files = [f for f in npy_files if not ('template' in f.name.lower())]
    npy_files.sort()
    
    for npy_file in npy_files:
        filename = npy_file.stem
        
        match = re.match(r'UN(\d+)_(org|lab)', filename)
        
        if not match:
            print(f"Skipping {npy_file.name}: Doesn't match expected pattern")
            continue
        
        case_num = match.group(1)
        file_type = match.group(2)
        
        if file_type == 'org':
            output_filename = f"wbone_{case_num}_0000.nii.gz"
            output_subdir = org_dir
        elif file_type == 'lab':
            output_filename = f"wbone_{case_num}.nii.gz"
            output_subdir = lab_dir
        else:
            print(f"Skipping {npy_file.name}: Unknown type")
            continue
        
        # Load numpy file
        try:
            data = np.load(npy_file)
            print(f"Loaded {npy_file.name} with shape: {data.shape}")
            
            affine = np.eye(4)
            nifti_img = nib.Nifti1Image(data, affine)
            
            output_path = os.path.join(output_subdir, output_filename)
            nib.save(nifti_img, output_path)
            
        except Exception as e:
            print(f"Error converting {npy_file.name}: {str(e)}")

if __name__ == "__main__":

    # for nnunet only, expects strict formatting

    args = sys.argv[1:]

    if len(args) != 2:
        print("Usage: python convert_numpy_to_nifti.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = args[0]
    output_dir = args[1]

    if not os.path.isdir(input_dir):
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)
    
    convert_npy_to_niigz(input_dir, output_dir)