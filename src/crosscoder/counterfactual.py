from typing import Dict, List, Tuple

import pandas as pd
import torch
import numpy as np

from . import config


def compute_counterfactual_sensitivity(
    feature_activations: Dict,
) -> pd.DataFrame:
    z_u = feature_activations["z_u"]
    z_c = feature_activations["z_c"]
    sample_ids = feature_activations["sample_ids"]
    image_types = feature_activations["image_types"]
    
    sample_to_indices = {}
    for idx, (sid, itype) in enumerate(zip(sample_ids, image_types)):
        if sid not in sample_to_indices:
            sample_to_indices[sid] = {}
        sample_to_indices[sid][itype] = idx
    
    paired_samples = []
    for sid, type_dict in sample_to_indices.items():
        if "original" in type_dict and "counterfact" in type_dict:
            paired_samples.append((sid, type_dict["original"], type_dict["counterfact"]))
    
    num_features = z_u.shape[1]
    cf_u = torch.zeros(num_features)
    cf_c = torch.zeros(num_features)
    
    for sid, orig_idx, cf_idx in paired_samples:
        z_u_orig = z_u[orig_idx]
        z_u_cf = z_u[cf_idx]
        z_c_orig = z_c[orig_idx]
        z_c_cf = z_c[cf_idx]
        
        cf_u += torch.abs(z_u_cf - z_u_orig)
        cf_c += torch.abs(z_c_cf - z_c_orig)
    
    num_pairs = len(paired_samples)
    cf_u /= num_pairs
    cf_c /= num_pairs
    
    cf_shift = cf_c - cf_u
    
    records = []
    for i in range(num_features):
        records.append({
            "feature_id": i,
            "cf_u": cf_u[i].item(),
            "cf_c": cf_c[i].item(),
            "cf_shift": cf_shift[i].item(),
        })
    
    return pd.DataFrame(records)


def classify_cf_level(cf_scores_df: pd.DataFrame, threshold_type: str = "median") -> pd.DataFrame:
    if threshold_type == "median":
        threshold_u = cf_scores_df["cf_u"].median()
        threshold_c = cf_scores_df["cf_c"].median()
    else:
        threshold_u = cf_scores_df["cf_u"].mean()
        threshold_c = cf_scores_df["cf_c"].mean()
    
    cf_scores_df = cf_scores_df.copy()
    cf_scores_df["cf_level_u"] = cf_scores_df["cf_u"].apply(
        lambda x: "high" if x > threshold_u else "low"
    )
    cf_scores_df["cf_level_c"] = cf_scores_df["cf_c"].apply(
        lambda x: "high" if x > threshold_c else "low"
    )
    cf_scores_df["cf_threshold_u"] = threshold_u
    cf_scores_df["cf_threshold_c"] = threshold_c
    
    return cf_scores_df


def merge_classification_with_cf(
    classification_df: pd.DataFrame,
    cf_scores_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = classification_df.merge(cf_scores_df, on="feature_id", how="left")
    return merged


def compute_cf_shift_by_class(merged_df: pd.DataFrame) -> Dict[str, Dict]:
    results = {}
    
    for primary_class in merged_df["primary_class"].unique():
        class_df = merged_df[merged_df["primary_class"] == primary_class]
        
        cf_shifts = class_df["cf_shift"].values
        cf_u_values = class_df["cf_u"].values
        cf_c_values = class_df["cf_c"].values
        
        results[primary_class] = {
            "count": len(class_df),
            "cf_shift_mean": float(np.mean(cf_shifts)),
            "cf_shift_std": float(np.std(cf_shifts)),
            "cf_shift_median": float(np.median(cf_shifts)),
            "cf_u_mean": float(np.mean(cf_u_values)),
            "cf_c_mean": float(np.mean(cf_c_values)),
            "high_cf_u_count": int((class_df["cf_level_u"] == "high").sum()),
            "low_cf_u_count": int((class_df["cf_level_u"] == "low").sum()),
            "high_cf_c_count": int((class_df["cf_level_c"] == "high").sum()),
            "low_cf_c_count": int((class_df["cf_level_c"] == "low").sum()),
        }
    
    return results


def identify_visual_evidence_features(merged_df: pd.DataFrame) -> Dict[str, List[int]]:
    uncompressed_only_high_cf = merged_df[
        (merged_df["primary_class"] == "uncompressed_only") & 
        (merged_df["cf_level_u"] == "high")
    ]["feature_id"].tolist()
    
    shared_redirected_cf_shift = merged_df[
        (merged_df["primary_class"] == "shared_redirected") & 
        (merged_df["cf_shift"] < 0)
    ]["feature_id"].tolist()
    
    compressed_only_high_cf = merged_df[
        (merged_df["primary_class"] == "compressed_only") & 
        (merged_df["cf_level_c"] == "high")
    ]["feature_id"].tolist()
    
    return {
        "lost_visual_evidence": uncompressed_only_high_cf,
        "redirected_visual_to_prior": shared_redirected_cf_shift,
        "new_compensatory_visual": compressed_only_high_cf,
    }


def compute_per_sample_feature_activations(
    feature_activations: Dict,
) -> Dict[str, Dict]:
    z_u = feature_activations["z_u"]
    z_c = feature_activations["z_c"]
    sample_ids = feature_activations["sample_ids"]
    image_types = feature_activations["image_types"]
    
    per_sample = {}
    for idx, (sid, itype) in enumerate(zip(sample_ids, image_types)):
        key = f"{sid}_{itype}"
        per_sample[key] = {
            "z_u": z_u[idx],
            "z_c": z_c[idx],
            "sample_id": sid,
            "image_type": itype,
        }
    
    return per_sample


def save_cf_results(cf_scores_df: pd.DataFrame, output_path: str) -> None:
    cf_scores_df.to_csv(output_path, index=False)


def load_cf_results(input_path: str) -> pd.DataFrame:
    return pd.read_csv(input_path)
