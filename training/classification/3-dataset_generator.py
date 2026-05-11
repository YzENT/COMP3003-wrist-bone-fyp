"""
After analysing dataset from step 2, save it as _cleaned_v3.csv, or whatever
For all fractured bones (patho label = 1), we select a healthy bone of same type
With small probability that another healthy bone type is selected to introduce variability
!!! remove cluster_voxel_count, somehow noise
"""

import pandas as pd
import numpy as np
import random
import os
import sys
from pathlib import Path

# config

# 1:1 = 42*2 = 84
# 2:1 = 42*3 = 126 samples total
HEALTHY_RATIO = 2 # 2 (healthy) : 1 (fracture)
OTHER_BONE_PROB = 0.05 # 0.05 = 5% chance other bone gets picked
RANDOM_SEED = 42 # yup chatgpt ahh seed

def extract_bone_type(dataset_id: str) -> str:
    """Extract bone type from dataset_id e.g. 'UN001_scaphoid' -> 'scaphoid'."""
    return dataset_id.split("_")[-1]

def pick_one_healthy(healthy_pool, selected_healthy_ids, target_bone,
                     all_bone_types, other_bone_prob, random_seed):
    """
    Pick a single healthy bone sample, preferring same bone type.
    Returns (dataset_id, chosen_type, reason) or None if pool exhausted.
    """
    if random.random() < other_bone_prob:
        other_types = [b for b in all_bone_types if b != target_bone]
        chosen_type = random.choice(other_types) if other_types else target_bone
        reason = f"random other type ({chosen_type})"
    else:
        chosen_type = target_bone
        reason = "same type"

    not_yet_picked = ~healthy_pool["dataset_id"].isin(selected_healthy_ids)

    candidates = healthy_pool[(healthy_pool["bone_type"] == chosen_type) & not_yet_picked]

    selected_id = candidates.sample(n=1, random_state=random_seed)["dataset_id"].values[0]
    return selected_id, chosen_type, reason

def select_samples(csv_path, output_path, healthy_ratio, other_bone_prob, random_seed):
    label_col = "pathology_label"
    id_col = "dataset_id"
    fracture_label = 1

    random.seed(random_seed)
    np.random.seed(random_seed)

    df = pd.read_csv(csv_path)

    print(f"\nFull dataset : {len(df)} samples")
    print(f"Label distribution:")
    for label, count in df[label_col].value_counts().sort_index().items():
        print(f"Label {label}: {count}")

    df["bone_type"] = df[id_col].apply(extract_bone_type)

    # extract all fractured bones
    fractured = df[df[label_col] == fracture_label].copy()
    print(f"\nFractured bones (label={fracture_label}): {len(fractured)}")
    print("Bone type breakdown:")
    for bone, count in fractured["bone_type"].value_counts().items():
        print(f"{bone:15s}: {count}")

    # get pool of available healthy bones
    healthy_pool = df[df[label_col] == 0].copy()
    print(f"\nHealthy bone pool available: {len(healthy_pool)}")
    print(f"Healthy bones needed: {len(fractured) * healthy_ratio}")

    # for each fractured, select HEALTHY_RATIO amount of bones
    selected_healthy_ids = set()
    selection_log = []

    all_bone_types = healthy_pool["bone_type"].unique().tolist()

    for _, frac_row in fractured.iterrows():
        target_bone = frac_row["bone_type"]

        for i in range(healthy_ratio):
            result = pick_one_healthy(
                healthy_pool, selected_healthy_ids, target_bone,
                all_bone_types, other_bone_prob, random_seed
            )

            selected_id, chosen_type, reason = result
            selected_healthy_ids.add(selected_id)

            selection_log.append({
                "fractured_id":     frac_row[id_col],
                "fractured_type":   target_bone,
                "healthy_id":       selected_id,
                "healthy_type":     chosen_type,
                "selection_reason": reason,
                "slot":             i + 1,
            })

    # combine fractured + selected healthy
    selected_healthy = healthy_pool[healthy_pool[id_col].isin(selected_healthy_ids)]
    final_df = pd.concat([fractured, selected_healthy], ignore_index=True)

    # sort by patient id, then bone type
    BONE_ORDER = [
        "ulnar", "radius", "scaphoid", "lunate", "triquetrum",
        "hamate", "trapezoid", "capitate", "trapezium", "pisiform"
    ]

    final_df["patient_id"] = final_df[id_col].apply(
        lambda x: "_".join(x.split("_")[:-1])
    )
    final_df["bone_order"] = final_df["bone_type"].apply(
        lambda b: BONE_ORDER.index(b) if b in BONE_ORDER else len(BONE_ORDER)
    )

    final_df = final_df.sort_values(
        by=["patient_id", "bone_order"]
    ).reset_index(drop=True)

    # Drop helper columns before saving
    final_df = final_df.drop(columns=["bone_type", "patient_id", "bone_order"])

    log_df = pd.DataFrame(selection_log)

    # Check for duplicates
    duplicate_count = final_df[id_col].duplicated().sum()
    print("Scanning for duplicate entries...")
    if duplicate_count > 0:
        print(f"WARNING: {duplicate_count} duplicate entries found!")
    else:
        print("No duplicate entries found.")

    output_csv  = output_path / "selected_samples.csv"
    summary_txt = output_path / "selection_summary.txt"

    final_df.to_csv(output_csv, index=False)
    print(f"\nSelected samples saved: {output_csv}")

    with open(summary_txt, "w") as f:
        f.write(log_df.to_string(index=False))
    print(f"Selection log saved: {summary_txt}")

    return final_df

if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) != 1:
        print("Usage: python 3-dataset_generator.py <input_csv_cleaned_v3>")
        sys.exit(1)

    input_csv = args[0]
    output_path = Path(input_csv).parent / "dataset generator script output"
    os.makedirs(output_path, exist_ok=True)

    final_df = select_samples(
        csv_path        = input_csv,
        output_path     = output_path,
        healthy_ratio   = HEALTHY_RATIO,
        other_bone_prob = OTHER_BONE_PROB,
        random_seed     = RANDOM_SEED,
    )