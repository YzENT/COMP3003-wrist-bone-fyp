import pydicom
import numpy as np
import os
import glob
from collections import Counter, defaultdict
from scipy.ndimage import zoom
import nibabel as nib
import sys

# according to dicom viewer (buggy and not stable)
TARGET_SERIES = "COR T1"
IGNORE_MRI_KEYWORDS = ["POSDISP", "_TIRM_", "REPEAT", "FSPGR"]

def normalize(volume):
    min_val = volume.min()
    max_val = volume.max()
    if max_val == min_val:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - min_val) / (max_val - min_val)).astype(np.float32)


def pick_cor_t1_series(series_dict):
    candidates = []

    for uid, slices in series_dict.items():
        try:
            desc = slices[0].SeriesDescription.strip()
        except AttributeError:
            desc = ""
        print(f"Found series: '{desc}' (uid ...{uid[-6:]})")

        desc_upper = desc.upper()

        # if desc_upper.startswith("POSDISP"):
        #     continue

        if any(k in desc_upper for k in IGNORE_MRI_KEYWORDS):
            continue

        if "COR" in desc_upper and "T1" in desc_upper:
            candidates.append((uid, desc))

    if len(candidates) == 1:
        uid, desc = candidates[0]
        print(f"Selected: '{desc}'")
        return uid
    elif len(candidates) == 0:
        return None
    else:
        descs = [f"'{d}'" for _, d in candidates]
        print(f"Error: ambiguous matches — expected 1 but found {len(candidates)}: {', '.join(descs)}")
        return None


# def pick_best_series(series_dict):
#     # pick the series with the highest normalized mean intensity (most content)
#     best_uid = None
#     best_score = -1

#     for uid, slices in series_dict.items():
#         arrays = [s.pixel_array for s in slices]
#         shapes = [a.shape for a in arrays]
#         most_common = Counter(shapes).most_common(1)[0][0]
#         arrays = [a for a in arrays if a.shape == most_common]

#         volume = np.stack(arrays, axis=0).astype(float)
#         max_val = volume.max()
#         if max_val == 0:
#             continue

#         score = (volume / max_val).mean() + (volume / max_val).std()
#         print(f"  Series {uid[-6:]}: shape={volume.shape}, score={score:.4f}")

#         if score > best_score:
#             best_score = score
#             best_uid = uid

#     return best_uid


def process_dicom_files(input_dir, output_dir, format, target_shape):
    # structure: input_dir/<patient_id>/<scan_date>/MR/*.dcm
    dcm_folders = []
    for patient_id in sorted(os.listdir(input_dir)):
        patient_dir = os.path.join(input_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue
        for scan_date in sorted(os.listdir(patient_dir)):
            scan_dir = os.path.join(patient_dir, scan_date)
            if not os.path.isdir(scan_dir):
                continue
            mr_dir = os.path.join(scan_dir, "MR")
            if os.path.isdir(mr_dir):
                dcm_folders.append((patient_id, scan_date, mr_dir))

    if not dcm_folders:
        print(f"No MR folders found under {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    processed_count = 0
    error_count = 0

    for patient_id, scan_date, mr_dir in dcm_folders:
        label = f"{patient_id}/{scan_date}/MR"

        try:
            print(f"Processing {label}...")

            dcm_files = glob.glob(os.path.join(mr_dir, "*.dcm"))
            if not dcm_files:
                print(f"Error: no .dcm files found in {mr_dir}")
                error_count += 1
                continue

            # Group by SeriesInstanceUID
            series_dict = defaultdict(list)
            for f in dcm_files:
                ds = pydicom.dcmread(f)
                series_dict[str(ds.SeriesInstanceUID)].append(ds)

            # Sort slices within each series
            for uid in series_dict:
                slices = series_dict[uid]
                try:
                    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
                except AttributeError:
                    try:
                        slices.sort(key=lambda s: int(s.InstanceNumber))
                    except AttributeError:
                        slices.sort(key=lambda s: s.filename)

            best_uid = pick_cor_t1_series(series_dict)
            # best_uid = pick_best_series(series_dict)
            if best_uid is None:
                print(f"Error: no '{TARGET_SERIES}' series found in {label}")
                error_count += 1
                continue

            best_slices = series_dict[best_uid]

            # Filter to consistent slice shape
            shapes = [s.pixel_array.shape for s in best_slices]
            most_common = Counter(shapes).most_common(1)[0][0]
            best_slices = [s for s in best_slices if s.pixel_array.shape == most_common]

            dicom_data = np.stack([s.pixel_array for s in best_slices], axis=0).astype(np.float32)
            # transpose to (H, W, D)
            dicom_data = dicom_data.transpose(1, 2, 0)

            if format == "npy":
                zoom_factors = [target_shape[i] / dicom_data.shape[i] for i in range(3)]
                dicom_resampled = normalize(zoom(dicom_data, zoom_factors, order=1))
                out_filename = f"UX{int(patient_id):03d}_org_dicom.npy"
                np.save(os.path.join(output_dir, out_filename), dicom_resampled)
                print(f"Saved: {out_filename} ({dicom_data.shape} -> {dicom_resampled.shape})")

            elif format == "niigz":
                zoom_factors = [target_shape[i] / dicom_data.shape[i] for i in range(3)]
                dicom_resampled = normalize(zoom(dicom_data, zoom_factors, order=1))
                out_filename = f"wbone_{int(patient_id):03d}_0000.nii.gz"
                nib.save(nib.Nifti1Image(dicom_resampled, affine=np.eye(4)), os.path.join(output_dir, out_filename))
                print(f"Saved: {out_filename} ({dicom_data.shape} -> {dicom_resampled.shape})")

            processed_count += 1

        except Exception as e:
            print(f"Error processing {label}: {e}")
            error_count += 1

    print(f"\nDone. Processed: {processed_count}, Errors: {error_count}")


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) < 3:
        print("Usage: python convert_dicom_to_usable.py <input_dir> <output_dir> --format=npy/niigz --res=XX YY ZZ")
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
            target_shape = (first, int(args[i+1]), int(args[i+2]))

    if target_shape is None:
        print("Error: --res=XX YY ZZ is required")
        sys.exit(1)

    process_dicom_files(input_dir, output_dir, format, target_shape)