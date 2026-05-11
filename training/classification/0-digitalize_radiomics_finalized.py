"""
Extracts radiomic features from the MRI dataset.
The data should be in nifti format ONLY. There should be 2 folders from the main nifti folder, 'lab' and 'org'
E.g: <nifti folder>\\org\\wbone_001_0000.nii.gz (corresponds UN001_org.npy)
E.g: <nifti folder>\\lab\\wbone_001.nii.gz (corresponds UN001_lab.npy)
The nifti namings are already set-up properly in previous scripts, just make sure training is based on 200x200x72 rather than 128x128x48.
Usage: python 0-digitalize_radiomics_finalized.py <input_dir> <output_dir>
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

# Global
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

# Erode bone sizes differently based on bone type
# E.g: 0.10 means erode by 10 percent
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

def extract_all_bones_features(niigz_folder, output_folder, pathology_labels=None):

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

    # DETERMINE
    print(f"Found {len(label_files)} patients to process")
    print(f"Expected total: {len(label_files) * len(BONE_LABELS)} bone samples")
    print("="*70)

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

        # Load nifti data
        scan_nii  = nib.load(str(mri_file))
        mask_nii  = nib.load(str(label_file))
        scan_data = scan_nii.get_fdata()
        mask_data = mask_nii.get_fdata()

        for bone_label, bone_name in BONE_LABELS.items():
            dataset_id = f"UN{patient_num}_{bone_name}"

            try:
                # run the pyradiomics extractor
                result = extractor.execute(
                    str(mri_file),
                    str(label_file),
                    label=bone_label
                )

                features = {'dataset_id': dataset_id}

                for key, value in result.items():
                    if not key.startswith('diagnostics_'):
                        features[key] = value

                # 3D Spatial Clustering features (ignore first)
                clustering_feats = extract_clustering_features(
                    scan_data=scan_data,
                    mask_data=mask_data,
                    bone_label=bone_label,
                    bone_name=bone_name
                )
                features.update(clustering_feats)

                if pathology_labels and (patient_num, bone_name) in pathology_labels:
                    features['pathology_label'] = pathology_labels[(patient_num, bone_name)]
                else:
                    features['pathology_label'] = 0  # no labels = auto assume = 0 = healthy

                all_features.append(features)
                total_processed += 1
                print(f"Done: {patient_num}_{bone_name}")

            except Exception as e:
                print(f"Error: {bone_name}: {str(e)}")
                continue

    if len(all_features) == 0:
        print("\nError: No features were extracted!")
        return None

    df = pd.DataFrame(all_features)

    metadata_cols   = ['dataset_id', 'pathology_label']
    clustering_cols = ['cluster_dark_ratio', 'cluster_intensity_contrast',
                       'cluster_relative_distance', 'cluster_dark_voxel_count']
    other_cols = [col for col in df.columns
                  if col not in metadata_cols and col not in clustering_cols]

    df = df[metadata_cols + other_cols + clustering_cols]

    # Save to CSV
    output_path = Path(output_folder) / "all_bones_features.csv"
    df.to_csv(output_path, index=False)

    print("\nPathology distribution:")
    print(f"Normal (0):   {(df['pathology_label']==0).sum()}")
    print(f"Fractured (1): {(df['pathology_label']==1).sum()}")
    print(f"Bruised (2):  {(df['pathology_label']==2).sum()}")

    return df

def extract_clustering_features(scan_data, mask_data, bone_label, bone_name):
    """
    Extract SPATIAL clustering features for a single bone fragment.

    Unlike intensity-only clustering, each voxel is represented as:
        [intensity, x, y, z]
    so K-Means only groups voxels together if they are BOTH dark AND
    physically close to each other in 3D space. This means scattered
    dark noise voxels are ignored, and only coherent dark regions
    (i.e. fracture lines) form their own cluster.

    This function:
      1. Isolates the bone fragment using its label
      2. Erodes the mask by a per-bone percentage to remove surface voxels
      3. Normalises intensity and spatial coordinates to equal scale
      4. Runs K-Means (k=2) on [intensity, x, y, z] combined
      5. Computes features describing the spatially coherent dark cluster

    Args:
        scan_data:  3D numpy array of MRI intensities
        mask_data:  3D numpy array of bone labels (from nnUNet)
        bone_label: Integer label of the bone to process
        bone_name:  String name of the bone (used to look up erosion percent)

    Returns:
        Dictionary with 4 clustering features, or None-filled dict if extraction fails
    """

    # Default return in case anything goes wrong
    empty_result = {
        'cluster_dark_ratio': None,
        'cluster_intensity_contrast': None,
        'cluster_relative_distance': None,
        'cluster_dark_voxel_count': None
    }

    try:
        # ── Step 1: Isolate this bone fragment ──────────────────────────────
        bone_mask = (mask_data == bone_label)

        if bone_mask.sum() == 0:
            return empty_result

        # ── Step 2: Erode mask to remove surface voxels ─────────────────────
        erosion_percent    = BONE_EROSION_PERCENT.get(bone_name, 0.05)
        erosion_iterations = compute_erosion_iterations(bone_mask, erosion_percent)
        eroded_mask        = binary_erosion(bone_mask, iterations=erosion_iterations)

        if eroded_mask.sum() == 0:
            eroded_mask = bone_mask

        # ── Step 3: Build combined [intensity, x, y, z] feature matrix ──────
        # Get 3D coordinates of all interior voxels
        interior_coords      = np.argwhere(eroded_mask)          # shape: (N, 3)
        interior_intensities = scan_data[eroded_mask]            # shape: (N,)

        if len(interior_intensities) < 10:
            return empty_result

        # Normalise intensity to [0, 1] so it's on the same scale as
        # the spatial coordinates (which are also normalised below).
        # Without this, intensity would be drowned out by large coord values.
        i_min, i_max = interior_intensities.min(), interior_intensities.max()
        if i_max > i_min:
            norm_intensity = (interior_intensities - i_min) / (i_max - i_min)
        else:
            norm_intensity = np.zeros_like(interior_intensities)

        # Normalise spatial coordinates to [0, 1] per axis
        coord_min  = interior_coords.min(axis=0)
        coord_max  = interior_coords.max(axis=0)
        coord_range = coord_max - coord_min
        coord_range[coord_range == 0] = 1  # avoid division by zero
        norm_coords = (interior_coords - coord_min) / coord_range  # shape: (N, 3)

        # Stack into (N, 4): each voxel = [intensity, x, y, z]
        # INTENSITY_WEIGHT controls how much intensity matters vs position.
        # Higher = clustering driven more by intensity (like before).
        # Lower  = clustering driven more by spatial proximity.

        # INTENSITY_WEIGHT = 1.5 # mean acc -> 0.7465, without voxel count
        INTENSITY_WEIGHT = 2.0  # mean acc -> 0.7698, without voxel count
        # INTENSITY_WEIGHT = 3.0  # mean acc -> 0.7542
        
        combined = np.column_stack([
            norm_intensity * INTENSITY_WEIGHT,
            norm_coords
        ])

        # ── Step 4: K-Means clustering on [intensity, x, y, z] ──────────────
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(combined)

        labels  = kmeans.labels_
        centers = kmeans.cluster_centers_   # shape: (2, 4)

        # Identify dark cluster by the intensity component of each centre
        # (first column of centers, before weighting was applied)
        dark_cluster   = int(np.argmin(centers[:, 0]))   # lower intensity = darker
        bright_cluster = int(np.argmax(centers[:, 0]))

        # ── Step 5: Compute clustering features ──────────────────────────────

        dark_voxel_count  = int(np.sum(labels == dark_cluster))
        total_voxel_count = len(labels)

        # Feature 1: dark_ratio
        # Proportion of interior voxels in the dark spatial cluster.
        # Fractured bone → higher ratio.
        dark_ratio = dark_voxel_count / total_voxel_count

        # Feature 2: intensity_contrast
        # Difference in mean ORIGINAL (un-normalised) intensity between clusters.
        dark_mean   = interior_intensities[labels == dark_cluster].mean()
        bright_mean = interior_intensities[labels == bright_cluster].mean()
        intensity_contrast = float(bright_mean - dark_mean)

        # ── Step 6: Centroid distance ─────────────────────────────────────────
        # Now uses the spatially coherent dark cluster's centroid, which is
        # more meaningful than before since the cluster is spatially contiguous.
        bone_centroid = interior_coords.mean(axis=0)
        dark_coords   = interior_coords[labels == dark_cluster]
        dark_centroid = dark_coords.mean(axis=0)

        centroid_distance    = float(np.linalg.norm(dark_centroid - bone_centroid))
        distances_from_centre = np.linalg.norm(interior_coords - bone_centroid, axis=1)
        bone_radius          = float(np.max(distances_from_centre))

        # Feature 3: relative_distance
        # ≈ 0.0 → dark cluster is at bone centre (suspicious for fracture)
        # ≈ 1.0 → dark cluster is at bone edge (likely surface artifact)
        relative_distance = centroid_distance / bone_radius if bone_radius > 0 else 0.0

        # Feature 4: dark_voxel_count
        return {
            'cluster_dark_ratio':         round(dark_ratio, 6),
            'cluster_intensity_contrast': round(intensity_contrast, 4),
            'cluster_relative_distance':  round(relative_distance, 6),
            'cluster_dark_voxel_count':   dark_voxel_count
        }

    except Exception as e:
        print(f"[clustering warning] {e}")
        return empty_result

if __name__ == "__main__":
    # niigz_folder = r"C:\npy_128_128_48\alr script test"
    # output_folder = r"C:\npy_128_128_48\alr script test\ballz"
    # output_csv = 'all_bones_features.csv'
    args = sys.argv[1:]

    if len(args) != 2:
        print("Usage: python 0-digitalize_radiomics_finalized.py <input_dir> <output_dir>")
        sys.exit(1)

    niigz_folder = args[0]
    output_folder = args[1]

    pathology_labels = {
        # 0 = normal
        # 1 = fractured
        # 2 = bruised (abondoned)
        # These ground truth obtained from Xin Chen
        ('006', 'ulnar'): 1,
        ('006', 'radius'): 1,
        ('007', 'scaphoid'): 2,
        ('009', 'trapezium'): 2,
        ('012', 'radius'): 2,
        ('013', 'scaphoid'): 1,
        ('017', 'scaphoid'): 1,
        ('019', 'lunate'): 2,
        ('021', 'radius'): 1,
        ('021', 'lunate'): 1,
        ('023', 'capitate'): 1,
        ('025', 'ulnar'): 2,
        ('025', 'hamate'): 1,
        ('026', 'triquetrum'): 2,
        ('027', 'radius'): 1,
        ('027', 'scaphoid'): 1,
        ('027', 'capitate'): 2,
        ('028', 'pisiform'): 2,
        ('029', 'radius'): 1,
        ('029', 'scaphoid'): 1,
        ('034', 'radius'): 1,
        ('040', 'radius'): 1,
        ('042', 'radius'): 1,
        ('042', 'scaphoid'): 1,
        ('044', 'scaphoid'): 1,
        ('045', 'scaphoid'): 1,
        ('054', 'radius'): 2,
        ('054', 'scaphoid'): 1,
        ('054', 'lunate'): 2,
        ('054', 'triquetrum'): 2,
        ('054', 'trapezoid'): 2,
        ('054', 'capitate'): 2,
        ('059', 'hamate'): 2,
        ('059', 'capitate'): 2,
        ('061', 'scaphoid'): 1,
        ('062', 'triquetrum'): 2,
        ('063', 'radius'): 1,
        ('066', 'radius'): 2,
        ('066', 'capitate'): 1,
        ('069', 'scaphoid'): 1,
        ('070', 'radius'): 1,
        ('072', 'scaphoid'): 1,
        ('073', 'radius'): 1,
        ('073', 'scaphoid'): 1,
        ('075', 'scaphoid'): 1,
        ('076', 'scaphoid'): 1,
        ('076', 'triquetrum'): 2,
        ('079', 'scaphoid'): 1,
        ('080', 'radius'): 1,
        ('081', 'scaphoid'): 1,
        ('081', 'lunate'): 2,
        ('081', 'hamate'): 2,
        ('081', 'capitate'): 1,
        ('086', 'trapezium'): 2,
        ('089', 'scaphoid'): 1,
        ('089', 'trapezoid'): 2,
        ('090', 'scaphoid'): 1,
        ('090', 'capitate'): 1,
        ('092', 'scaphoid'): 1,
        ('093', 'triquetrum'): 1,
        ('095', 'radius'): 2,
        ('095', 'scaphoid'): 1,
        ('095', 'lunate'): 2,
        ('095', 'triquetrum'): 2,
        ('098', 'ulnar'): 2,
        ('098', 'radius'): 1,
        ('098', 'lunate'): 2,
        ('101', 'scaphoid'): 1,
        ('101', 'trapezium'): 1
    }

    df = extract_all_bones_features(
        niigz_folder=niigz_folder,
        output_folder=output_folder,
        pathology_labels=pathology_labels
    )