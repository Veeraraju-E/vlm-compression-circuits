from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config


plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    "uncompressed_only": "#E74C3C",
    "compressed_only": "#3498DB",
    "shared_aligned": "#2ECC71",
    "shared_redirected": "#9B59B6",
    "shared_attenuated": "#F39C12",
    "shared_intermediate": "#95A5A6",
    "other": "#7F8C8D",
}


def plot_loss_curves(training_history: Dict, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = training_history["epochs"]
    
    ax = axes[0, 0]
    ax.plot(epochs, training_history["train_loss"], label="Train", linewidth=2)
    ax.plot(epochs, training_history["val_loss"], label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(epochs, training_history["self_recon"], label="Self-Recon", linewidth=2)
    ax.plot(epochs, training_history["cross_recon"], label="Cross-Recon", linewidth=2)
    ax.plot(epochs, training_history["sparsity"], label="Sparsity", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Component")
    ax.set_title("Loss Components")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(epochs, training_history["val_fve_u"], label="FVE_u", linewidth=2)
    ax.plot(epochs, training_history["val_fve_c"], label="FVE_c", linewidth=2)
    ax.axhline(y=config.FVE_THRESHOLD, color='r', linestyle='--', label=f"Threshold ({config.FVE_THRESHOLD})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FVE")
    ax.set_title("Fraction of Variance Explained")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(epochs, training_history["dead_neurons"], label="Dead Neurons", linewidth=2, color="red")
    ax.axhline(y=config.DEAD_NEURON_THRESHOLD, color='orange', linestyle='--', 
               label=f"Threshold ({config.DEAD_NEURON_THRESHOLD})")
    ax2 = ax.twinx()
    ax2.plot(epochs, training_history["l0_u"], label="L0_u", linewidth=2, color="blue", alpha=0.7)
    ax2.plot(epochs, training_history["l0_c"], label="L0_c", linewidth=2, color="green", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dead Neuron Fraction", color="red")
    ax2.set_ylabel("L0 Sparsity", color="blue")
    ax.set_title("Sparsity Metrics")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_rho_histogram(classification_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rho_values = classification_df["rho"].values
    
    ax.hist(rho_values, bins=50, range=(0, 1), edgecolor='black', alpha=0.7, color='steelblue')
    
    ax.axvline(x=config.RHO_UNCOMPRESSED_ONLY, color='red', linestyle='--', 
               label=f"Uncompressed-only threshold ({config.RHO_UNCOMPRESSED_ONLY})")
    ax.axvline(x=config.RHO_COMPRESSED_ONLY, color='blue', linestyle='--',
               label=f"Compressed-only threshold ({config.RHO_COMPRESSED_ONLY})")
    ax.axvline(x=config.RHO_SHARED_LOW, color='green', linestyle=':', alpha=0.7,
               label=f"Shared range ({config.RHO_SHARED_LOW}-{config.RHO_SHARED_HIGH})")
    ax.axvline(x=config.RHO_SHARED_HIGH, color='green', linestyle=':', alpha=0.7)
    
    ax.set_xlabel("ρ (Decoder Norm Ratio)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Feature Decoder Norm Ratios (ρ)", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_rho_theta_scatter(classification_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for class_name, color in COLORS.items():
        class_df = classification_df[classification_df["primary_class"] == class_name]
        if len(class_df) > 0:
            ax.scatter(class_df["rho"], class_df["theta"], 
                      c=color, label=f"{class_name} ({len(class_df)})", 
                      alpha=0.6, s=20)
    
    ax.axvline(x=config.RHO_UNCOMPRESSED_ONLY, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=config.RHO_COMPRESSED_ONLY, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=config.RHO_SHARED_LOW, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=config.RHO_SHARED_HIGH, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=config.THETA_ALIGNED, color='gray', linestyle='-.', alpha=0.5)
    ax.axhline(y=config.THETA_REDIRECTED, color='gray', linestyle='-.', alpha=0.5)
    
    ax.set_xlabel("ρ (Decoder Norm Ratio)", fontsize=12)
    ax.set_ylabel("θ (Decoder Cosine Similarity)", fontsize=12)
    ax.set_title("Feature Classification: (ρ, θ) Space", fontsize=14)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cf_distribution_per_class(merged_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    class_order = ["uncompressed_only", "compressed_only", "shared_aligned", 
                   "shared_redirected", "shared_attenuated"]
    
    plot_df = merged_df[merged_df["primary_class"].isin(class_order)]
    
    ax = axes[0]
    if "cf_u" in plot_df.columns:
        sns.boxplot(data=plot_df, x="primary_class", y="cf_u", order=class_order,
                   palette=COLORS, ax=ax)
        ax.set_xlabel("Feature Class", fontsize=12)
        ax.set_ylabel("CF_u (Uncompressed)", fontsize=12)
        ax.set_title("Counterfactual Sensitivity (Uncompressed Model)", fontsize=14)
        ax.tick_params(axis='x', rotation=45)
    
    ax = axes[1]
    if "cf_c" in plot_df.columns:
        sns.boxplot(data=plot_df, x="primary_class", y="cf_c", order=class_order,
                   palette=COLORS, ax=ax)
        ax.set_xlabel("Feature Class", fontsize=12)
        ax.set_ylabel("CF_c (Compressed)", fontsize=12)
        ax.set_title("Counterfactual Sensitivity (Compressed Model)", fontsize=14)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cf_shift_per_class(merged_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    class_order = ["uncompressed_only", "compressed_only", "shared_aligned", 
                   "shared_redirected", "shared_attenuated"]
    
    if "cf_shift" not in merged_df.columns:
        plt.close()
        return
    
    shift_by_class = merged_df.groupby("primary_class")["cf_shift"].agg(["mean", "std"])
    shift_by_class = shift_by_class.reindex(class_order).dropna()
    
    colors = [COLORS.get(c, "#7F8C8D") for c in shift_by_class.index]
    
    bars = ax.bar(range(len(shift_by_class)), shift_by_class["mean"], 
                  yerr=shift_by_class["std"], color=colors, edgecolor='black',
                  capsize=5, alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_xticks(range(len(shift_by_class)))
    ax.set_xticklabels(shift_by_class.index, rotation=45, ha='right')
    ax.set_xlabel("Feature Class", fontsize=12)
    ax.set_ylabel("CF Shift (CF_c - CF_u)", fontsize=12)
    ax.set_title("Counterfactual Sensitivity Shift by Feature Class", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_superposition_analysis(superposition_results: Dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    features = superposition_results.get("features", {})
    if not features:
        plt.close()
        return
    
    r2_values = [f["r2"] for f in features.values()]
    n_nonzero_values = [f["n_nonzero"] for f in features.values()]
    is_superposition = [f["is_superposition"] for f in features.values()]
    
    ax = axes[0]
    colors = ['green' if s else 'red' for s in is_superposition]
    ax.hist(r2_values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=config.SUPERPOSITION_R2_THRESHOLD, color='red', linestyle='--',
               label=f"R² threshold ({config.SUPERPOSITION_R2_THRESHOLD})")
    ax.set_xlabel("R² Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Superposition Analysis: R² Distribution", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.scatter(r2_values, n_nonzero_values, c=colors, alpha=0.6, s=30)
    ax.axhline(y=config.SUPERPOSITION_MAX_CONSTITUENTS, color='orange', linestyle='--',
               label=f"Max constituents ({config.SUPERPOSITION_MAX_CONSTITUENTS})")
    ax.axvline(x=config.SUPERPOSITION_R2_THRESHOLD, color='red', linestyle='--',
               label=f"R² threshold")
    ax.set_xlabel("R² Score", fontsize=12)
    ax.set_ylabel("Number of Non-zero Constituents", fontsize=12)
    ax.set_title("Superposition: R² vs Constituent Count", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_fsr_comparison(metrics_dict: Dict[str, Dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    configs = list(metrics_dict.keys())
    fsr_values = [m.get("feature_sharing_ratio", 0) for m in metrics_dict.values()]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(configs)))
    
    bars = ax.bar(range(len(configs)), fsr_values, color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Feature Sharing Ratio (FSR)", fontsize=12)
    ax.set_title("Feature Sharing Ratio Comparison Across Configurations", fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, fsr_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_method_comparison(
    wanda_metrics: Dict,
    awq_metrics: Dict,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_to_compare = ["feature_sharing_ratio", "semantic_stability_score", "superposition_fraction"]
    titles = ["Feature Sharing Ratio", "Semantic Stability Score", "Superposition Fraction"]
    
    for ax, metric, title in zip(axes, metrics_to_compare, titles):
        wanda_val = wanda_metrics.get(metric, 0)
        awq_val = awq_metrics.get(metric, 0)
        
        bars = ax.bar(["Wanda", "AWQ"], [wanda_val, awq_val], 
                     color=["#E74C3C", "#3498DB"], edgecolor='black', alpha=0.8)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(0, max(wanda_val, awq_val) * 1.2 + 0.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, [wanda_val, awq_val]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_class_distribution(classification_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    class_counts = classification_df["primary_class"].value_counts()
    
    colors = [COLORS.get(c, "#7F8C8D") for c in class_counts.index]
    
    bars = ax.bar(range(len(class_counts)), class_counts.values, 
                  color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_xticks(range(len(class_counts)))
    ax.set_xticklabels(class_counts.index, rotation=45, ha='right')
    ax.set_xlabel("Feature Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Feature Class Distribution", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, class_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               str(val), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_all_plots(
    training_history: Dict,
    classification_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    superposition_results: Dict,
    plots_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_loss_curves(training_history, plots_dir / "loss_curves.png")
    plot_rho_histogram(classification_df, plots_dir / "rho_histogram.png")
    plot_rho_theta_scatter(classification_df, plots_dir / "rho_theta_scatter.png")
    plot_cf_distribution_per_class(merged_df, plots_dir / "cf_distribution_per_class.png")
    plot_cf_shift_per_class(merged_df, plots_dir / "cf_shift_per_class.png")
    plot_superposition_analysis(superposition_results, plots_dir / "superposition_analysis.png")
    plot_class_distribution(classification_df, plots_dir / "class_distribution.png")
