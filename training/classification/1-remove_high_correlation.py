"""
Remove highly correlated features from dataset, threshold is 0.95.
Expects a _cleaned.csv version input, where you manually remove zero variance beforehand
Usage: python 1-remove_high_correlation.py <input_csv_cleaned>
Create a folder called 'high correlation script output'
Then we output _cleaned_v2.csv
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os

def remove_correlated_features(csv_path, correlation_threshold, output_path):
    output_csv = output_path / "all_bones_features_cleaned_v2.csv"
    df = pd.read_csv(csv_path)
    
    metadata_cols = ['dataset_id', 'pathology_label']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    print(f"Original feature count: {len(feature_cols)}")

    X = df[feature_cols]
    corr_matrix = X.corr().abs()
    
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = []
    dropped_pairs = []
    
    for column in upper_triangle.columns:
        correlated_features = upper_triangle.index[upper_triangle[column] > correlation_threshold].tolist()
        
        if correlated_features:
            for corr_feature in correlated_features:
                if corr_feature not in to_drop:
                    to_drop.append(corr_feature)
                    corr_value = corr_matrix.loc[column, corr_feature]
                    dropped_pairs.append((column, corr_feature, corr_value))
    
    if dropped_pairs:
        print(f"Found {len(to_drop)} highly correlated features to remove:\n")
        for kept, dropped, corr in dropped_pairs:
            print(f"Keeping: {kept}")
            print(f"Dropping: {dropped}")
            print(f"Correlation: {corr:.4f}")
            print()
    else:
        print(f"No features found with correlation > {correlation_threshold}")
    
    features_to_keep = [col for col in feature_cols if col not in to_drop]
    df_cleaned = df[metadata_cols + features_to_keep]
    
    print(f"Original features: {len(feature_cols)}")
    print(f"Removed features: {len(to_drop)}")
    print(f"Remaining features: {len(features_to_keep)}")
    print(f"Reduction: {len(to_drop)/len(feature_cols)*100:.1f}%")
    
    df_cleaned.to_csv(output_csv, index=False)
    print(f"\nCleaned dataset saved to: {output_csv}")
    
    # heatmap copy pasted
    heatmap_path = output_path / 'correlation_heatmap_cleaned.png'
    plt.figure(figsize=(12, 10))
    corr_cleaned = df_cleaned[features_to_keep].corr()
    sns.heatmap(corr_cleaned, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'Feature Correlation Heatmap (After Removing Correlation > {correlation_threshold})')
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    
    return df_cleaned


if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) != 1:
        print("Usage: python 1-remove_high_correlation.py <input_csv_cleaned>")
        sys.exit(1)

    input_csv = args[0]

    correlation_threshold = 0.95
    
    input_path = Path(input_csv)
    output_path = Path(input_csv).parent / "high correlation script output"

    os.makedirs(output_path, exist_ok=True)
    
    df_cleaned = remove_correlated_features(
        csv_path=input_csv,
        correlation_threshold=correlation_threshold,
        output_path=output_path
    )
    
    print("\nProcess complete.")