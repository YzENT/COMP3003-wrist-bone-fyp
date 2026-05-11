"""
Extracts radiomic features MRI scans
No pathology labels, script is for model prediction input only
The data should be in nifti format ONLY (200x200x72). There should be 2 folders from the main nifti folder, 'lab' and 'org'
E.g: <nifti folder>\\org\\wbone_001_0000.nii.gz (corresponds UN001_org.npy)
E.g: <nifti folder>\\lab\\wbone_001.nii.gz (corresponds UN001_lab.npy)
Only the columns used during training are kept (INFERENCE_COLUMNS)
Usage: python 0-generate_inference_spreadsheet.py <input_dir> <output_dir>
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from radiomics import featureextractor
from sklearn.cluster import KMeans
from scipy.ndimage import binary_erosion
import logging
import warnings
import nibabel as nib
import sys

# SHHHH
logging.getLogger('radiomics').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

INFERENCE_COLUMNS = [
    'dataset_id',
    'original_firstorder_Skewness',
    'original_firstorder_TotalEnergy',
    'original_glrlm_RunPercentage',
    'original_glrlm_RunVariance',
    'original_glrlm_ShortRunLowGrayLevelEmphasis',
    'original_glszm_LargeAreaLowGrayLevelEmphasis',
    'original_glszm_SmallAreaLowGrayLevelEmphasis',
    'original_glszm_ZonePercentage',
    'cluster_dark_ratio',
]

# ── Globals ────────────────────────────────────────────────────────────────────
BONE_LABELS = {
    1: 'ulnar',
    2: 'radius',
    3: 'scaphoid',
    4: 'lunate',
    5: 'triquetrum',
    6: 'hamate',
    7: 'trapezoid',
    8: 'capitate',
    9: 'trapezium',
    10: 'pisiform'
}

BONE_EROSION_PERCENT = {
    'ulnar':      0.10,
    'radius':     0.15,
    'scaphoid':   0.08,
    'lunate':     0.08,
    'triquetrum': 0.08,
    'hamate':     0.08,
    'trapezoid':  0.05,
    'capitate':   0.09,
    'trapezium':  0.05,
    'pisiform':   0.05,
}


def compute_erosion_iterations(bone_mask, erosion_percent):
    voxel_count = bone_mask.sum()
    if voxel_count == 0:
        return 1

    # assuming bone is roughly spherical: V = (4/3) * pi * r^3
    estimated_radius = (voxel_count * 3 / (4 * np.pi)) ** (1 / 3)

    # Convert percentage of radius to number of voxels (minimum 1)
    iterations = max(1, int(round(estimated_radius * erosion_percent)))
    return iterations


def extract_clustering_features(scan_data, mask_data, bone_label, bone_name):
    empty_result = {
        'cluster_dark_ratio': None,
        'cluster_intensity_contrast': None,
        'cluster_relative_distance': None,
        'cluster_dark_voxel_count': None
    }

    try:
        bone_mask = (mask_data == bone_label)
        if bone_mask.sum() == 0:
            return empty_result

        erosion_percent = BONE_EROSION_PERCENT.get(bone_name, 0.05)
        erosion_iterations = compute_erosion_iterations(bone_mask, erosion_percent)
        eroded_mask = binary_erosion(bone_mask, iterations=erosion_iterations)

        if eroded_mask.sum() == 0:
            eroded_mask = bone_mask

        interior_coords = np.argwhere(eroded_mask)
        interior_intensities = scan_data[eroded_mask]

        if len(interior_intensities) < 10:
            return empty_result

        i_min, i_max = interior_intensities.min(), interior_intensities.max()
        norm_intensity = (interior_intensities - i_min) / (i_max - i_min) if i_max > i_min else np.zeros_like(interior_intensities)

        coord_min = interior_coords.min(axis=0)
        coord_max = interior_coords.max(axis=0)
        coord_range = coord_max - coord_min
        coord_range[coord_range == 0] = 1
        norm_coords = (interior_coords - coord_min) / coord_range

        INTENSITY_WEIGHT = 2.0
        combined = np.column_stack([norm_intensity * INTENSITY_WEIGHT, norm_coords])

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(combined)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        dark_cluster = int(np.argmin(centers[:, 0]))
        bright_cluster = int(np.argmax(centers[:, 0]))

        dark_voxel_count = int(np.sum(labels == dark_cluster))
        total_voxel_count = len(labels)
        dark_ratio = dark_voxel_count / total_voxel_count # only need up to here but whatever, just keep rest bah

        dark_mean = interior_intensities[labels == dark_cluster].mean()
        bright_mean = interior_intensities[labels == bright_cluster].mean()
        intensity_contrast = float(bright_mean - dark_mean)

        bone_centroid = interior_coords.mean(axis=0)
        dark_centroid = interior_coords[labels == dark_cluster].mean(axis=0)
        centroid_distance = float(np.linalg.norm(dark_centroid - bone_centroid))
        distances_from_centre = np.linalg.norm(interior_coords - bone_centroid, axis=1)
        bone_radius = float(np.max(distances_from_centre))
        relative_distance = centroid_distance / bone_radius if bone_radius > 0 else 0.0

        return {
            'cluster_dark_ratio':         round(dark_ratio, 6),
            'cluster_intensity_contrast': round(intensity_contrast, 4),
            'cluster_relative_distance':  round(relative_distance, 6),
            'cluster_dark_voxel_count':   dark_voxel_count
        }

    except Exception as e:
        print(f"[clustering warning] {e}")
        return empty_result


def run_inference(niigz_folder, output_folder):
    niigz_path = Path(niigz_folder)
    lab_folder = niigz_path / 'lab'
    org_folder = niigz_path / 'org'
    os.makedirs(output_folder, exist_ok=True)

    if not lab_folder.exists() or not org_folder.exists():
        print(f"Error: 'lab' or 'org' folder not found in {niigz_folder}")
        return

    label_files = sorted(lab_folder.glob('wbone_*.nii.gz'))
    if len(label_files) == 0:
        print(f"Error: No wbone_*.nii.gz files found in {lab_folder}")
        return

    print(f"Found {len(label_files)} patients to process")
    print(f"Expected total: {len(label_files) * len(BONE_LABELS)} bone samples")
    print("=" * 70)

    extractor = featureextractor.RadiomicsFeatureExtractor()
    all_features = []
    total_processed = 0

    for label_file in label_files:
        patient_num = label_file.name.replace('wbone_', '').replace('.nii.gz', '')
        mri_file = org_folder / f"wbone_{patient_num}_0000.nii.gz"

        if not mri_file.exists():
            print(f"Warning: MRI file not found for patient {patient_num}, skipping...")
            continue

        print(f"\nProcessing patient {patient_num}...")

        scan_nii = nib.load(str(mri_file))
        mask_nii = nib.load(str(label_file))
        scan_data = scan_nii.get_fdata()
        mask_data = mask_nii.get_fdata()

        for bone_label, bone_name in BONE_LABELS.items():
            dataset_id = f"UX{patient_num}_{bone_name}"

            try:
                result = extractor.execute(str(mri_file), str(label_file), label=bone_label)
                features = {'dataset_id': dataset_id}

                for key, value in result.items():
                    if not key.startswith('diagnostics_'):
                        features[key] = value

                clustering_feats = extract_clustering_features(
                    scan_data=scan_data,
                    mask_data=mask_data,
                    bone_label=bone_label,
                    bone_name=bone_name
                )
                features.update(clustering_feats)

                all_features.append(features)
                total_processed += 1
                print(f"Done: {patient_num}_{bone_name}")

            except Exception as e:
                print(f"Error [{bone_name}]: {str(e)}")
                continue

    if len(all_features) == 0:
        print("\nError: No features were extracted!")
        return None

    df = pd.DataFrame(all_features)

    missing_cols = [col for col in INFERENCE_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"\nWarning: The following INFERENCE_COLUMNS were not found and will be skipped: {missing_cols}")

    valid_cols = [col for col in INFERENCE_COLUMNS if col in df.columns]
    df = df[valid_cols]

    output_path = Path(output_folder) / "inference_features.csv"
    df.to_csv(output_path, index=False)

    print(f"\nDone. {total_processed} bone samples written to: {output_path}")

    return df


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) != 2:
        print("Usage: python inference_radiomics.py <input_dir> <output_dir>")
        sys.exit(1)

    niigz_folder  = args[0]
    output_folder = args[1]

    run_inference(
        niigz_folder=niigz_folder,
        output_folder=output_folder
    )