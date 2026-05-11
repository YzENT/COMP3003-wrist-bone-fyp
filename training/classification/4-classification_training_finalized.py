import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, FunctionTransformer,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# config

# True = Healthy vs Fractured
# False = Healthy vs Fractured vs Bruised (RIP)
USE_BINARY = True

NORMALISERS = ["standard", "minmax", "robust", "maxabs", "log", "mean"]

# Cross-validation
N_FOLDS = 5
N_ITER = 20 # hyperparameter combinations to try per model
INNER_CV = 3  # inner CV folds during hyperparameter search

RANDOM_STATE = 42 # mmhm..

class MeanNormaliser:
    # mean normalisation: (X - train_mean) / (train_max - train_min)
    def fit(self, X, y=None):
        X = np.array(X)
        self.train_mean  = X.mean(axis=0)
        self.train_range = X.max(axis=0) - X.min(axis=0)
        self.train_range[self.train_range == 0] = 1  # avoid division by zero
        return self

    def transform(self, X, y=None):
        return (np.array(X) - self.train_mean) / self.train_range

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

# scalar/model factories
def get_scaler(scaler_name: str):
    scaler_options = {
        "standard": StandardScaler(),
        "minmax":   MinMaxScaler(),
        "robust":   RobustScaler(),
        "maxabs":   MaxAbsScaler(),
        "log":      FunctionTransformer(
                        func=lambda X: np.log1p(np.abs(X)) * np.sign(X),
                        validate=True,
                    ),
        "mean":     MeanNormaliser(),
    }
    if scaler_name not in scaler_options:
        raise ValueError(
            f"Unknown scaler '{scaler_name}'. Choose from: {list(scaler_options.keys())}"
        )
    return scaler_options[scaler_name]


def get_models(random_state=42, n_iter=20, inner_cv=3):
    """Return a dict of model_name -> RandomizedSearchCV-wrapped estimator."""

    def make_search(estimator, param_grid):
        return RandomizedSearchCV(
            estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=inner_cv,
            scoring="f1_weighted",
            refit=True,
            random_state=random_state,
            n_jobs=-1,
            error_score=0,
        )

    return {
        "Random Forest": make_search(
            RandomForestClassifier(class_weight="balanced", random_state=random_state),
            {
                "n_estimators":      [100, 200, 300],
                "max_depth":         [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf":  [1, 2, 4],
                "max_features":      ["sqrt", "log2"],
            },
        ),
        "Extra Trees": make_search(
            ExtraTreesClassifier(class_weight="balanced", random_state=random_state),
            {
                "n_estimators":      [100, 200, 300],
                "max_depth":         [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf":  [1, 2, 4],
                "max_features":      ["sqrt", "log2"],
            },
        ),
        "Gradient Boosting": make_search(
            GradientBoostingClassifier(random_state=random_state),
            {
                "n_estimators":     [100, 200, 300],
                "max_depth":        [3, 4, 6],
                "learning_rate":    [0.01, 0.05, 0.1, 0.2],
                "subsample":        [0.7, 0.8, 1.0],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "Logistic Regression": make_search(
            LogisticRegression(
                class_weight="balanced", max_iter=10000, random_state=random_state
            ),
            {
                "C":       [0.01, 0.1, 1.0, 10.0, 100.0],
                "penalty": ["l2"],
                "solver":  ["lbfgs", "saga"],
            },
        ),
        # some library update broke ts
        # "SVM (RBF)": make_search(
        #     SVC(class_weight="balanced", probability=True, random_state=random_state),
        #     {
        #         "C":      [0.1, 1.0, 10.0, 100.0],
        #         "gamma":  ["scale", "auto", 0.001, 0.01, 0.1],
        #         "kernel": ["rbf", "poly"],
        #     },
        # ),
        "SVM (RBF)": make_search(
            SVC(
                class_weight="balanced",
                probability=True,
                random_state=random_state,
            ),
            {
                "C":      [0.1, 1.0, 10.0],
                "gamma":  ["scale", "auto", 0.01],
                "kernel": ["rbf"],
            },
        ),
        "KNN": make_search(
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 9, 11, 15],
                "weights":     ["uniform", "distance"],
                "metric":      ["euclidean", "manhattan", "minkowski"],
            },
        ),
        "MLP": make_search(
            MLPClassifier(max_iter=500, early_stopping=True, random_state=random_state),
            {
                "hidden_layer_sizes": [
                    (64,), (128,), (64, 64), (128, 64), (128, 128), (256, 128, 64)
                ],
                "activation":         ["relu", "tanh"],
                "alpha":              [0.0001, 0.001, 0.01],
                "learning_rate_init": [0.001, 0.01],
            },
        ),
    }


def extract_patient_id(dataset_id: str) -> str:
    """Extract patient ID from a dataset ID — e.g. 'UN001_scaphoid' -> 'UN001'."""
    return "_".join(dataset_id.split("_")[:-1])


def save_mispredictions(all_dataset_ids, all_true_labels, all_predicted_labels, class_names, safe_model_name, normaliser_output_dir):
    misprediction_records = []
    for dataset_id, true_label, predicted_label in zip(all_dataset_ids, all_true_labels, all_predicted_labels):
        if true_label != predicted_label:
            if true_label == 0 and predicted_label == 1:
                error_type = "False Positive (predicted Pathological, actually Healthy)"
            elif true_label == 1 and predicted_label == 0:
                error_type = "False Negative (predicted Healthy, actually Pathological)"
            else:
                error_type = (
                    f"Misclassified ({class_names[true_label]} → {class_names[predicted_label]})"
                )

            misprediction_records.append({
                "dataset_id": dataset_id,
                "bone_type":  dataset_id.split("_")[-1],
                "patient_id": extract_patient_id(dataset_id),
                "true_label": class_names[true_label],
                "predicted":  class_names[predicted_label],
                "error_type": error_type,
            })

    mispred_df = pd.DataFrame(misprediction_records)

    total_samples = len(all_true_labels)
    n_errors = len(mispred_df)
    print(f"\nMispredictions : {n_errors}/{total_samples} ({n_errors / total_samples * 100:.1f}% error rate)")
    mispred_path = normaliser_output_dir / f"mispredictions_{safe_model_name}.csv"
    mispred_df.to_csv(mispred_path, index=False)
    print(f"Misprediction report saved: {mispred_path}")

    return mispred_df


def _plot_comparison(summary_df, scaler_name, output_folder):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bar_positions = range(len(summary_df))
    model_labels  = summary_df["model"].tolist()

    for ax, metric_col, std_col, bar_colour in [
        (axes[0], "mean_accuracy", "std_accuracy", "steelblue"),
        (axes[1], "mean_f1",       "std_f1",       "coral"),
    ]:
        bars = ax.bar(
            bar_positions, summary_df[metric_col],
            yerr=summary_df[std_col],
            color=bar_colour, alpha=0.8, capsize=5,
        )
        ax.set_xticks(list(bar_positions))
        ax.set_xticklabels(model_labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric_col.replace("mean_", "").capitalize())
        ax.set_title(
            f"[{scaler_name}] CV {metric_col.replace('mean_', '').capitalize()} by Model"
        )
        for bar, metric_value in zip(bars, summary_df[metric_col]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{metric_value:.3f}", ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()
    chart_path = output_folder / "model_comparison.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Model comparison chart saved: {chart_path}")


def _save_feature_importances(trained_models, feature_columns, scaler_name, output_folder):
    for model_name, fitted_model in trained_models.items():
        estimator = (
            fitted_model.best_estimator_
            if hasattr(fitted_model, "best_estimator_")
            else fitted_model
        )
        if not hasattr(estimator, "feature_importances_"):
            continue

        importances_df = pd.DataFrame({
            "feature":    feature_columns,
            "importance": estimator.feature_importances_,
        }).sort_values("importance", ascending=False)

        safe_model_name = (
            model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        )
        importances_path = output_folder / f"feature_importances_{safe_model_name}.csv"
        importances_df.to_csv(importances_path, index=False)


def _plot_cross_normaliser(combined_results_df, output_folder):
    model_names = combined_results_df["model"].unique()
    normaliser_names = combined_results_df["normaliser"].unique()
    bar_positions = np.arange(len(model_names))
    bar_width = 0.13
    bar_colours = ["steelblue", "coral", "seagreen", "orchid", "goldenrod", "tomato"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, metric_col, chart_title in [
        (axes[0], "mean_f1",       "Mean F1-Score by Model & Normaliser"),
        (axes[1], "mean_accuracy", "Mean Accuracy by Model & Normaliser"),
    ]:
        for offset, (normaliser, bar_colour) in enumerate(zip(normaliser_names, bar_colours)):
            normaliser_rows = combined_results_df[combined_results_df["normaliser"] == normaliser]
            metric_values = [
                normaliser_rows[normaliser_rows["model"] == model_name][metric_col].values[0]
                if len(normaliser_rows[normaliser_rows["model"] == model_name]) > 0 else 0
                for model_name in model_names
            ]
            std_col = metric_col.replace("mean_", "std_")
            std_values = [
                normaliser_rows[normaliser_rows["model"] == model_name][std_col].values[0]
                if len(normaliser_rows[normaliser_rows["model"] == model_name]) > 0 else 0
                for model_name in model_names
            ]
            ax.bar(
                bar_positions + offset * bar_width, metric_values, bar_width,
                yerr=std_values, label=normaliser, color=bar_colour, alpha=0.8, capsize=3,
            )

        ax.set_xticks(bar_positions + bar_width * (len(normaliser_names) / 2))
        ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(chart_title)
        ax.legend(title="Normaliser", fontsize=7)

    plt.tight_layout()
    chart_path = output_folder / "cross_normaliser_comparison.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Cross-normaliser chart saved: {chart_path}")


def plot_confusion_matrix(confusion_mat, class_names, title, save_path):
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        confusion_mat, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curves(fold_roc_data, model_name, scaler_name, class_names, save_path):
    is_binary = len(class_names) == 2

    fig, ax = plt.subplots(figsize=(7, 6))
    interpolated_tprs = []
    fold_aucs = []
    mean_fpr = np.linspace(0, 1, 200)

    for fold_index, (fpr, tpr) in enumerate(fold_roc_data):
        fold_auc = auc(fpr, tpr)
        fold_aucs.append(fold_auc)
        interpolated_tprs.append(np.interp(mean_fpr, fpr, tpr))
        interpolated_tprs[-1][0] = 0.0
        ax.plot(fpr, tpr, lw=1, alpha=0.35,
                label=f"Fold {fold_index + 1} (AUC = {fold_auc:.3f})")

    mean_tpr = np.mean(interpolated_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc  = np.std(fold_aucs)

    ax.plot(mean_fpr, mean_tpr, color="navy", lw=2.5,
            label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})")

    std_tpr = np.std(interpolated_tprs, axis=0)
    ax.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        alpha=0.15, color="navy", label="± 1 std. dev.",
    )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random chance")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    roc_label = "Binary" if is_binary else "OvR macro"
    ax.set_title(f"ROC Curve ({roc_label}) - {model_name} [{scaler_name}]")
    ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def run_one_normaliser(feature_matrix, target_labels, patient_groups, dataset_ids, class_names, scaler_name, n_folds, n_iter, inner_cv, random_state, output_folder):
    normaliser_output_dir = output_folder / scaler_name
    normaliser_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"NORMALISER: {scaler_name.upper()}")
    print(f"{'#' * 70}")

    cross_validator = GroupKFold(n_splits=n_folds)
    models = get_models(random_state, n_iter=n_iter, inner_cv=inner_cv)
    summary_rows = []
    all_mispreds = [] # collect mispredictions across all models for summary

    for model_name, model in models.items():
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_name}  |  Normaliser: {scaler_name}")
        print(f"{'=' * 70}")

        fold_metrics   = []
        all_true_labels, all_predicted_labels, all_sample_ids = [], [], []
        fold_roc_data  = []  # (fpr, tpr) per fold for ROC plotting

        for fold_num, (train_idx, val_idx) in enumerate(
            cross_validator.split(feature_matrix, target_labels, groups=patient_groups), 1
        ):
            X_train = feature_matrix.iloc[train_idx]
            X_val   = feature_matrix.iloc[val_idx]
            y_train = target_labels.iloc[train_idx]
            y_val   = target_labels.iloc[val_idx]

            scaler = get_scaler(scaler_name)
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model.fit(X_train_scaled, y_train)

            if fold_num == 1:
                print(f"Best params: {model.best_params_}")

            y_pred = model.predict(X_val_scaled)

            # Collect per-fold ROC data
            if hasattr(model, "predict_proba"):
                predicted_probabilities = model.predict_proba(X_val_scaled)
                if predicted_probabilities.shape[1] == 2:  # binary
                    fpr, tpr, _ = roc_curve(y_val, predicted_probabilities[:, 1])
                else:  # multiclass — micro-average OvR
                    from sklearn.preprocessing import label_binarize
                    y_val_binarised = label_binarize(
                        y_val, classes=list(range(predicted_probabilities.shape[1]))
                    )
                    fpr_per_class, tpr_per_class = [], []
                    for class_idx in range(predicted_probabilities.shape[1]):
                        class_fpr, class_tpr, _ = roc_curve(
                            y_val_binarised[:, class_idx],
                            predicted_probabilities[:, class_idx],
                        )
                        fpr_per_class.append(class_fpr)
                        tpr_per_class.append(class_tpr)
                    # micro-average
                    fpr, tpr, _ = roc_curve(
                        y_val_binarised.ravel(), predicted_probabilities.ravel()
                    )
                fold_roc_data.append((fpr, tpr))

            fold_accuracy = accuracy_score(y_val, y_pred)
            fold_precision, fold_recall, fold_f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average="weighted", zero_division=0
            )

            val_patients = np.unique(patient_groups[val_idx])
            print(
                f"Fold {fold_num}/{n_folds}  |  "
                f"Acc: {fold_accuracy:.3f}  P: {fold_precision:.3f}  "
                f"R: {fold_recall:.3f}  F1: {fold_f1:.3f}  "
                f"|  Validation Count: {len(val_patients)}"
            )

            fold_metrics.append({
                "fold":      fold_num,
                "accuracy":  fold_accuracy,
                "precision": fold_precision,
                "recall":    fold_recall,
                "f1":        fold_f1,
            })
            all_true_labels.extend(y_val.tolist())
            all_predicted_labels.extend(y_pred.tolist())
            all_sample_ids.extend(dataset_ids.iloc[val_idx].tolist())

        # Aggregate metrics across folds
        fold_metrics_df = pd.DataFrame(fold_metrics)
        mean_accuracy = fold_metrics_df["accuracy"].mean()
        mean_f1 = fold_metrics_df["f1"].mean()

        print(f"\nMean Accuracy: {mean_accuracy:.3f} (+/- {fold_metrics_df['accuracy'].std():.3f})")
        print(f"Mean Precision: {fold_metrics_df['precision'].mean():.3f} (+/- {fold_metrics_df['precision'].std():.3f})")
        print(f"Mean Recall: {fold_metrics_df['recall'].mean():.3f} (+/- {fold_metrics_df['recall'].std():.3f})")
        print(f"Mean F1-Score: {mean_f1:.3f} (+/- {fold_metrics_df['f1'].std():.3f})")

        # Confusion matrix
        confusion_mat = confusion_matrix(all_true_labels, all_predicted_labels)
        safe_model_name = (
            model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        )
        cm_save_path = normaliser_output_dir / f"confusion_matrix_{safe_model_name}.png"
        plot_confusion_matrix(
            confusion_mat, class_names,
            title=f"Confusion Matrix - {model_name} [{scaler_name}] (All Folds)",
            save_path=cm_save_path,
        )
        print(f"\nConfusion matrix saved: {cm_save_path}")

        # ROC curve (per fold + mean)
        if fold_roc_data:
            roc_save_path = normaliser_output_dir / f"roc_curve_{safe_model_name}.png"
            plot_roc_curves(
                fold_roc_data, model_name, scaler_name, class_names, roc_save_path
            )
            print(f"ROC curve saved: {roc_save_path}")

        # Misprediction report
        mispred_df = save_mispredictions(
            all_sample_ids, all_true_labels, all_predicted_labels,
            class_names, safe_model_name, normaliser_output_dir,
        )
        if len(mispred_df) > 0:
            mispred_df["model"] = model_name
            mispred_df["normaliser"] = scaler_name
            all_mispreds.append(mispred_df)

        summary_rows.append({
            "model":          model_name,
            "normaliser":     scaler_name,
            "mean_accuracy":  round(mean_accuracy, 4),
            "mean_precision": round(fold_metrics_df["precision"].mean(), 4),
            "mean_recall":    round(fold_metrics_df["recall"].mean(), 4),
            "mean_f1":        round(mean_f1, 4),
            "std_accuracy":   round(fold_metrics_df["accuracy"].std(), 4),
            "std_f1":         round(fold_metrics_df["f1"].std(), 4),
        })

    # Per-normaliser summary CSV + chart
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_f1", ascending=False)
    summary_path = normaliser_output_dir / "model_comparison.csv"
    summary_df.to_csv(summary_path, index=False)

    _plot_comparison(summary_df, scaler_name, normaliser_output_dir)
    _save_feature_importances(models, feature_matrix, scaler_name, normaliser_output_dir)

    # get commonly mispredicted for this normaliser
    if all_mispreds:
        combined_mispreds_df = pd.concat(all_mispreds, ignore_index=True)
        combined_mispreds_path = normaliser_output_dir / "mispredictions_all_models.csv"
        combined_mispreds_df.to_csv(combined_mispreds_path, index=False)

        print(f"\n  --- Misprediction Summary [{scaler_name}] ---")
        print("Most frequently misclassified bones across all models:")
        mispred_freq = (
            combined_mispreds_df.groupby("dataset_id")
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        for sample_id, error_count in mispred_freq.items():
            bone_type = sample_id.split("_")[-1]
            print(
                f"{sample_id:25s} ({bone_type:12s}): misclassified by {error_count} model(s)"
            )
        print(f"Consolidated misprediction report saved: {combined_mispreds_path}")

    return summary_df, models

def train_and_evaluate(csv_path, output_path, normalisers, use_binary, n_folds, n_iter, inner_cv, random_state):
    output_path = Path(output_path)
    raw_df = pd.read_csv(csv_path)

    dataset_ids = raw_df["dataset_id"]
    patient_groups = np.array(dataset_ids.apply(extract_patient_id))

    feature_matrix = raw_df.drop(columns=["dataset_id", "pathology_label"])

    if use_binary:
        target_labels = (raw_df["pathology_label"] > 0).astype(int)
        class_names = ["Healthy", "Pathological"]
        print("\nClassification: BINARY (Healthy vs Pathological)")
    else:
        target_labels = raw_df["pathology_label"]
        class_names = ["Healthy", "Fractured", "Bruised"]
        print("\nClassification: 3-CLASS (Healthy / Fractured / Bruised)")

    all_normaliser_summaries = []
    all_trained_models = {}  # keyed by (scaler_name, model_name)

    for scaler_name in normalisers:
        normaliser_summary_df, trained_models = run_one_normaliser(
            feature_matrix, target_labels, patient_groups, dataset_ids,
            class_names, scaler_name, n_folds, n_iter, inner_cv,
            random_state, output_path,
        )
        all_normaliser_summaries.append(normaliser_summary_df)
        for model_name, fitted_model in trained_models.items():
            all_trained_models[(scaler_name, model_name)] = fitted_model

    # Cross-normaliser comparison
    combined_results_df = pd.concat(all_normaliser_summaries, ignore_index=True)
    combined_sorted_df  = combined_results_df.sort_values(
        ["mean_f1", "std_f1"], ascending=[False, True]
    )

    combined_results_path = output_path / "all_normalisers_comparison.csv"
    combined_results_df.to_csv(combined_results_path, index=False)

    best_result_row = combined_sorted_df.iloc[0]
    print(f"\nBest overall : {best_result_row['model']} with {best_result_row['normaliser']} normaliser")
    print(f"Mean Accuracy: {best_result_row['mean_accuracy']:.4f} (+/- {best_result_row['std_accuracy']:.4f})")
    print(f"Mean F1: {best_result_row['mean_f1']:.4f} (+/- {best_result_row['std_f1']:.4f})")

    # Refit the best model on the full dataset and save
    best_model_name = best_result_row["model"]
    best_scaler_name = best_result_row["normaliser"]
    best_fitted_model = all_trained_models[(best_scaler_name, best_model_name)]

    final_scaler = get_scaler(best_scaler_name)
    full_feature_scaled = final_scaler.fit_transform(feature_matrix)
    best_fitted_model.fit(full_feature_scaled, target_labels)

    safe_best_model_name = (
        best_model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    )
    model_save_path  = output_path / f"best_model_{safe_best_model_name}_{best_scaler_name}.pkl"
    scaler_save_path = output_path / f"best_scaler_{best_scaler_name}.pkl"

    joblib.dump(best_fitted_model, model_save_path)
    joblib.dump(final_scaler, scaler_save_path)

    print(f"\nBest model saved: {model_save_path}")
    print(f"Scaler saved: {scaler_save_path}")

    _plot_cross_normaliser(combined_results_df, output_path)

    print("TRAINING COMPLETE!")

    return combined_sorted_df


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) != 2:
        print("Usage: python 4-classification_training.py <selected_csv_path> <output_dir>")
        sys.exit(1)

    csv_path = args[0]
    output_folder = args[1]

    os.makedirs(output_folder, exist_ok=True)

    summary = train_and_evaluate(
        csv_path     = csv_path,
        output_path  = output_folder,
        normalisers  = NORMALISERS,
        use_binary   = USE_BINARY,
        n_folds      = N_FOLDS,
        n_iter       = N_ITER,
        inner_cv     = INNER_CV,
        random_state = RANDOM_STATE,
    )