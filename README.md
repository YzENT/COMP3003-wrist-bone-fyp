# Code Submissions

## Installation

The following libraries need to be installed before using this project:

- nnunetv2
- scikit-learn
- pyradiomics (create a separate environment; Python 3.9.25 is recommended)
- nibabel
- pandas, numpy, matplotlib
- pydicom
- napari

## Training

### Segmentation

1. Convert numpy data to nifti format
2. Place files in nnunet's directory (refer to nnunet's documentation for setup)
3. Run the nnunetv2 train command
4. Use the included nifti to numpy conversion script for observation

### Classification

1. Activate the environment previously set up for PyRadiomics
2. Convert the files into a spreadsheet format
3. Run the files in `code submissions/training/classification` in the labeled order

## Inference

### Segmentation

1. Convert DICOM files into nifti format (auto-resized)
2. Run the nnunet model (refer to example file provided on how to execute. Link to download trained models for this project: https://uniofnottm-my.sharepoint.com/:u:/g/personal/hcyyc8_nottingham_ac_uk/IQAwMLxVZCsmRL9kI8NpPosXAW7gIbmiQayH9hb9bP1eyzU?e=yv0pWj)
3. Obtain masks and upscale them using `upscale_predicted_nifti_masks.py` to 200x200x72

### Classification

1. Use the environment set up for PyRadiomics
2. Generate inference spreadsheet (with filtered columns)
3. Generate predicted spreadsheet (refer to file naming conventions for execution order)