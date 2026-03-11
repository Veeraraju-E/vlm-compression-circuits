from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F

from . import config
from .model import SPARCCrossCoder


def compute_decoder_norm_ratio(W_u_dec: torch.Tensor, W_c_dec: torch.Tensor) -> torch.Tensor:
    W_u_norms = W_u_dec.norm(dim=0)
    W_c_norms = W_c_dec.norm(dim=0)
    rho = W_c_norms / (W_u_norms + W_c_norms + 1e-8)
    return rho


def compute_decoder_cosine_similarity(W_u_dec: torch.Tensor, W_c_dec: torch.Tensor) -> torch.Tensor:
    W_u_normalized = F.normalize(W_u_dec, dim=0)
    W_c_normalized = F.normalize(W_c_dec, dim=0)
    theta = (W_u_normalized * W_c_normalized).sum(dim=0)
    return theta


def classify_feature(rho: float, theta: float) -> str:
    if rho < config.RHO_UNCOMPRESSED_ONLY:
        return "uncompressed_only"
    elif rho > config.RHO_COMPRESSED_ONLY:
        return "compressed_only"
    elif config.RHO_SHARED_LOW < rho < config.RHO_SHARED_HIGH:
        if theta > config.THETA_ALIGNED:
            return "shared_aligned"
        elif theta < config.THETA_REDIRECTED:
            return "shared_redirected"
        else:
            return "shared_intermediate"
    elif config.RHO_UNCOMPRESSED_ONLY <= rho <= config.RHO_SHARED_LOW:
        return "shared_attenuated"
    else:
        return "other"


def classify_all_features(crosscoder: SPARCCrossCoder) -> pd.DataFrame:
    decoder_weights = crosscoder.get_decoder_weights()
    W_u_dec = decoder_weights["W_u_dec"]
    W_c_dec = decoder_weights["W_c_dec"]
    
    rho = compute_decoder_norm_ratio(W_u_dec, W_c_dec)
    theta = compute_decoder_cosine_similarity(W_u_dec, W_c_dec)
    
    W_u_norms = W_u_dec.norm(dim=0)
    W_c_norms = W_c_dec.norm(dim=0)
    
    num_features = rho.shape[0]
    
    records = []
    for i in range(num_features):
        rho_i = rho[i].item()
        theta_i = theta[i].item()
        primary_class = classify_feature(rho_i, theta_i)
        
        is_forced_shared = i in crosscoder.forced_shared_indices.tolist()
        
        records.append({
            "feature_id": i,
            "rho": rho_i,
            "theta": theta_i,
            "primary_class": primary_class,
            "W_u_dec_norm": W_u_norms[i].item(),
            "W_c_dec_norm": W_c_norms[i].item(),
            "is_forced_shared": is_forced_shared,
        })
    
    return pd.DataFrame(records)


def get_feature_class_counts(classification_df: pd.DataFrame) -> Dict[str, int]:
    return classification_df["primary_class"].value_counts().to_dict()


def get_features_by_class(classification_df: pd.DataFrame, feature_class: str) -> List[int]:
    return classification_df[classification_df["primary_class"] == feature_class]["feature_id"].tolist()


def compute_rho_histogram_data(classification_df: pd.DataFrame, num_bins: int = 50) -> Dict:
    rho_values = classification_df["rho"].values
    hist, bin_edges = torch.histogram(torch.tensor(rho_values), bins=num_bins, range=(0.0, 1.0))
    return {
        "counts": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "bin_centers": [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)],
    }


def compute_threshold_sensitivity(
    classification_df: pd.DataFrame,
    perturbation: float = 0.05
) -> Dict:
    original_counts = get_feature_class_counts(classification_df)
    
    rho_values = classification_df["rho"].values
    theta_values = classification_df["theta"].values
    
    perturbed_counts = {}
    
    for delta in [-perturbation, perturbation]:
        adjusted_rho_u = config.RHO_UNCOMPRESSED_ONLY + delta
        adjusted_rho_c = config.RHO_COMPRESSED_ONLY - delta
        
        counts = {
            "uncompressed_only": 0,
            "compressed_only": 0,
            "shared_aligned": 0,
            "shared_redirected": 0,
            "shared_intermediate": 0,
            "shared_attenuated": 0,
            "other": 0,
        }
        
        for rho, theta in zip(rho_values, theta_values):
            if rho < adjusted_rho_u:
                counts["uncompressed_only"] += 1
            elif rho > adjusted_rho_c:
                counts["compressed_only"] += 1
            elif config.RHO_SHARED_LOW < rho < config.RHO_SHARED_HIGH:
                if theta > config.THETA_ALIGNED:
                    counts["shared_aligned"] += 1
                elif theta < config.THETA_REDIRECTED:
                    counts["shared_redirected"] += 1
                else:
                    counts["shared_intermediate"] += 1
            elif adjusted_rho_u <= rho <= config.RHO_SHARED_LOW:
                counts["shared_attenuated"] += 1
            else:
                counts["other"] += 1
        
        perturbed_counts[f"delta_{delta:+.2f}"] = counts
    
    return {
        "original": original_counts,
        "perturbed": perturbed_counts,
        "perturbation": perturbation,
    }


def save_classification_results(
    classification_df: pd.DataFrame,
    output_path: str,
) -> None:
    classification_df.to_csv(output_path, index=False)


def load_classification_results(input_path: str) -> pd.DataFrame:
    return pd.read_csv(input_path)
