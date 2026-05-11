import numpy as np
import nibabel as nib
import os
from pathlib import Path
import re
import sys

"""
- Convert nnUNet prediction from nifti back to numpy.
- For visualization purposes
"""

def convert_niigz_to_npy(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    input_path = Path(input_dir)
    niigz_files = list(input_path.glob("*.nii.gz"))
    
    # ignore non-nifti files
    niigz_files = [f for f in niigz_files if f.suffix == '.gz']
    niigz_files.sort()
    
    for niigz_file in niigz_files:
        filename = niigz_file.name.replace('.nii.gz', '')
        
        # Extract case number from filename (wbone_047 -> 047)
        match = re.match(r'wbone_(\d+)', filename)
        
        if not match:
            print(f"Skipping {niigz_file.name}: Doesn't match expected pattern")
            continue
        
        case_num = match.group(1)
        output_filename = f"UN{case_num}_pred.npy"
        
        # Load nifti
        try:
            nifti_img = nib.load(niigz_file)
            data = nifti_img.get_fdata()
            
            print(f"Loaded {niigz_file.name} with shape: {data.shape}")
            
            # Save as numpy
            output_path = os.path.join(output_dir, output_filename)
            np.save(output_path, data)
            
            print(f"Converted: {niigz_file.name} -> {output_filename}")
            
        except Exception as e:
            print(f"Error converting {niigz_file.name}: {str(e)}")

if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) != 2:
        print("Usage: python convert_nifti_to_numpy.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = args[0]
    output_dir = args[1]

    if not os.path.isdir(input_dir):
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)
    
    convert_niigz_to_npy(input_dir, output_dir)