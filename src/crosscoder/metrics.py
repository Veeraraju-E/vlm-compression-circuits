from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config


def compute_feature_sharing_ratio(classification_df: pd.DataFrame) -> float:
    shared_classes = ["shared_aligned", "shared_redirected", "shared_intermediate", "shared_attenuated"]
    exclusive_classes = ["uncompressed_only", "compressed_only"]
    
    n_shared = classification_df[classification_df["primary_class"].isin(shared_classes)].shape[0]
    n_exclusive = classification_df[classification_df["primary_class"].isin(exclusive_classes)].shape[0]
    
    if n_shared + n_exclusive == 0:
        return 0.0
    
    return n_shared / (n_shared + n_exclusive)


def compute_semantic_stability_score(classification_df: pd.DataFrame) -> float:
    shared_mask = (
        (classification_df["rho"] > config.RHO_SHARED_LOW) & 
        (classification_df["rho"] < config.RHO_SHARED_HIGH)
    )
    shared_features = classification_df[shared_mask]
    
    if len(shared_features) == 0:
        return 0.0
    
    return shared_features["theta"].mean()


def compute_counterfactual_sensitivity_shift(merged_df: pd.DataFrame) -> Dict[str, float]:
    results = {}
    
    for primary_class in merged_df["primary_class"].unique():
        class_df = merged_df[merged_df["primary_class"] == primary_class]
        if len(class_df) > 0 and "cf_shift" in class_df.columns:
            results[primary_class] = class_df["cf_shift"].mean()
    
    return results


def compute_superposition_fraction(superposition_results: Dict) -> float:
    return superposition_results.get("superposition_fraction", 0.0)


def compute_all_primary_metrics(
    classification_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    superposition_results: Dict,
    training_history: Dict,
) -> Dict:
    fsr = compute_feature_sharing_ratio(classification_df)
    sss = compute_semantic_stability_score(classification_df)
    css = compute_counterfactual_sensitivity_shift(merged_df)
    sf = compute_superposition_fraction(superposition_results)
    
    fve_u = training_history["val_fve_u"][-1] if training_history["val_fve_u"] else 0.0
    fve_c = training_history["val_fve_c"][-1] if training_history["val_fve_c"] else 0.0
    dead_neurons = training_history["dead_neurons"][-1] if training_history["dead_neurons"] else 0.0
    l0_u = training_history["l0_u"][-1] if training_history["l0_u"] else 0.0
    l0_c = training_history["l0_c"][-1] if training_history["l0_c"] else 0.0
    
    class_counts = classification_df["primary_class"].value_counts().to_dict()
    
    return {
        "feature_sharing_ratio": fsr,
        "semantic_stability_score": sss,
        "counterfactual_sensitivity_shift": css,
        "superposition_fraction": sf,
        "fve_u": fve_u,
        "fve_c": fve_c,
        "dead_neuron_fraction": dead_neurons,
        "l0_sparsity_u": l0_u,
        "l0_sparsity_c": l0_c,
        "class_counts": class_counts,
        "total_features": len(classification_df),
    }


def test_hypothesis_h1(
    wanda_classification: pd.DataFrame,
    awq_classification: pd.DataFrame,
) -> Dict:
    wanda_u_only = (wanda_classification["primary_class"] == "uncompressed_only").sum()
    wanda_c_only = (wanda_classification["primary_class"] == "compressed_only").sum()
    wanda_attenuated = (wanda_classification["primary_class"] == "shared_attenuated").sum()
    
    awq_u_only = (awq_classification["primary_class"] == "uncompressed_only").sum()
    awq_c_only = (awq_classification["primary_class"] == "compressed_only").sum()
    awq_attenuated = (awq_classification["primary_class"] == "shared_attenuated").sum()
    
    wanda_exclusive = wanda_u_only + wanda_c_only
    awq_exclusive = awq_u_only + awq_c_only
    
    return {
        "wanda_exclusive_count": int(wanda_exclusive),
        "awq_exclusive_count": int(awq_exclusive),
        "wanda_attenuated_count": int(wanda_attenuated),
        "awq_attenuated_count": int(awq_attenuated),
        "hypothesis_supported": awq_attenuated > wanda_attenuated,
        "description": "H1: Wanda produces discrete feature loss, AWQ produces gradual attenuation",
    }


def test_hypothesis_h2(merged_df: pd.DataFrame) -> Dict:
    u_only = merged_df[merged_df["primary_class"] == "uncompressed_only"]
    
    if len(u_only) == 0 or "cf_level_u" not in u_only.columns:
        return {
            "high_cf_count": 0,
            "low_cf_count": 0,
            "ratio": 0.0,
            "hypothesis_supported": False,
            "description": "H2: Visual-evidence features disproportionately lost",
        }
    
    high_cf = (u_only["cf_level_u"] == "high").sum()
    low_cf = (u_only["cf_level_u"] == "low").sum()
    
    ratio = high_cf / (high_cf + low_cf) if (high_cf + low_cf) > 0 else 0.0
    
    return {
        "high_cf_count": int(high_cf),
        "low_cf_count": int(low_cf),
        "ratio": float(ratio),
        "hypothesis_supported": ratio > 0.5,
        "description": "H2: Visual-evidence features disproportionately lost",
    }


def test_hypothesis_h3(
    wanda_superposition: Dict,
    awq_superposition: Dict,
) -> Dict:
    wanda_sf = wanda_superposition.get("superposition_fraction", 0.0)
    awq_sf = awq_superposition.get("superposition_fraction", 0.0)
    
    return {
        "wanda_sf": float(wanda_sf),
        "awq_sf": float(awq_sf),
        "hypothesis_supported": wanda_sf > 0.5 and wanda_sf > awq_sf,
        "description": "H3: Wanda superposition fraction > 50% and > AWQ",
    }


def test_hypothesis_h4(
    cls_classification: pd.DataFrame,
    patch_classification: pd.DataFrame,
) -> Dict:
    fsr_cls = compute_feature_sharing_ratio(cls_classification)
    fsr_patch = compute_feature_sharing_ratio(patch_classification)
    
    return {
        "fsr_cls": float(fsr_cls),
        "fsr_patch": float(fsr_patch),
        "hypothesis_supported": fsr_cls > fsr_patch,
        "description": "H4: CLS tokens have higher FSR than patch tokens",
    }


def test_hypothesis_h5(
    v_classification: pd.DataFrame,
    p_classification: pd.DataFrame,
) -> Dict:
    v_redirected = (v_classification["primary_class"] == "shared_redirected").sum()
    p_redirected = (p_classification["primary_class"] == "shared_redirected").sum()
    
    return {
        "v_redirected_count": int(v_redirected),
        "p_redirected_count": int(p_redirected),
        "hypothesis_supported": p_redirected > v_redirected,
        "description": "H5: Projector has more shared-redirected features than vision encoder",
    }


def test_hypothesis_h6(p_merged_df: pd.DataFrame) -> Dict:
    redirected = p_merged_df[p_merged_df["primary_class"] == "shared_redirected"]
    
    if len(redirected) == 0 or "cf_shift" not in redirected.columns:
        return {
            "mean_cf_shift": 0.0,
            "negative_shift_count": 0,
            "total_redirected": 0,
            "hypothesis_supported": False,
            "description": "H6: Projector redirected features shift from visual to prior",
        }
    
    mean_shift = redirected["cf_shift"].mean()
    negative_count = (redirected["cf_shift"] < 0).sum()
    
    return {
        "mean_cf_shift": float(mean_shift),
        "negative_shift_count": int(negative_count),
        "total_redirected": len(redirected),
        "hypothesis_supported": mean_shift < 0,
        "description": "H6: Projector redirected features shift from visual to prior",
    }


def test_hypothesis_h7(
    v_metrics: Dict,
    p_metrics: Dict,
    vp_metrics: Dict,
) -> Dict:
    fsr_v = v_metrics.get("feature_sharing_ratio", 0.0)
    fsr_p = p_metrics.get("feature_sharing_ratio", 0.0)
    fsr_vp = vp_metrics.get("feature_sharing_ratio", 0.0)
    
    product = fsr_v * fsr_p
    
    return {
        "fsr_v": float(fsr_v),
        "fsr_p": float(fsr_p),
        "fsr_vp": float(fsr_vp),
        "fsr_v_times_p": float(product),
        "hypothesis_supported": fsr_vp < product,
        "description": "H7: FSR(V+P) is sub-additive: FSR(V+P) < FSR(V) × FSR(P)",
    }


def test_hypothesis_h8(
    blip_p_metrics: Dict,
    qwen3vl_p_metrics: Dict,
) -> Dict:
    fsr_blip = blip_p_metrics.get("feature_sharing_ratio", 0.0)
    fsr_qwen3vl = qwen3vl_p_metrics.get("feature_sharing_ratio", 0.0)
    
    return {
        "fsr_blip_p": float(fsr_blip),
        "fsr_qwen3vl_p": float(fsr_qwen3vl),
        "hypothesis_supported": fsr_blip > fsr_qwen3vl,
        "description": "H8: BLIP cross-attention has higher FSR than Qwen3-VL-2B projector",
    }


def compile_all_hypothesis_results(hypothesis_tests: Dict) -> pd.DataFrame:
    records = []
    for h_name, result in hypothesis_tests.items():
        records.append({
            "hypothesis": h_name,
            "supported": result.get("hypothesis_supported", False),
            "description": result.get("description", ""),
            **{k: v for k, v in result.items() if k not in ["hypothesis_supported", "description"]},
        })
    return pd.DataFrame(records)


def save_metrics(metrics: Dict, output_path: str) -> None:
    import json
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(input_path: str) -> Dict:
    import json
    with open(input_path) as f:
        return json.load(f)
