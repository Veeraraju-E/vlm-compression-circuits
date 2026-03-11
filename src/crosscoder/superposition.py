from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Lasso
from tqdm import tqdm

from . import config
from .model import SPARCCrossCoder


def get_top_activating_samples(
    z_c: torch.Tensor,
    feature_id: int,
    top_k: int = config.SUPERPOSITION_TOP_SAMPLES,
) -> List[int]:
    feature_activations = z_c[:, feature_id]
    _, top_indices = torch.topk(feature_activations, min(top_k, len(feature_activations)))
    return top_indices.tolist()


def fit_sparse_regression(
    W_c_dec_feature: np.ndarray,
    W_u_dec: np.ndarray,
    alpha: float = 0.01,
    max_iter: int = 10000,
) -> Tuple[np.ndarray, float, int]:
    lasso = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=False)
    lasso.fit(W_u_dec, W_c_dec_feature)
    
    coefficients = lasso.coef_
    
    y_pred = W_u_dec @ coefficients
    ss_res = np.sum((W_c_dec_feature - y_pred) ** 2)
    ss_tot = np.sum((W_c_dec_feature - np.mean(W_c_dec_feature)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    n_nonzero = np.sum(np.abs(coefficients) > 1e-6)
    
    return coefficients, r2, n_nonzero


def analyze_superposition_for_feature(
    feature_id: int,
    W_c_dec: torch.Tensor,
    W_u_dec: torch.Tensor,
    z_c: torch.Tensor,
    z_u: torch.Tensor,
    alpha: float = 0.01,
) -> Dict:
    W_c_dec_feature = W_c_dec[:, feature_id].cpu().numpy()
    W_u_dec_np = W_u_dec.cpu().numpy()
    
    coefficients, r2, n_nonzero = fit_sparse_regression(
        W_c_dec_feature, W_u_dec_np, alpha=alpha
    )
    
    is_superposition = (
        r2 > config.SUPERPOSITION_R2_THRESHOLD and 
        n_nonzero <= config.SUPERPOSITION_MAX_CONSTITUENTS and
        n_nonzero >= 2
    )
    
    nonzero_indices = np.where(np.abs(coefficients) > 1e-6)[0]
    constituent_features = [
        {"feature_id": int(idx), "weight": float(coefficients[idx])}
        for idx in nonzero_indices
    ]
    constituent_features.sort(key=lambda x: abs(x["weight"]), reverse=True)
    
    top_samples = get_top_activating_samples(z_c, feature_id)
    
    return {
        "feature_id": feature_id,
        "r2": float(r2),
        "n_nonzero": int(n_nonzero),
        "is_superposition": is_superposition,
        "constituent_features": constituent_features,
        "top_activating_samples": top_samples[:10],
    }


def analyze_all_compressed_only_features(
    crosscoder: SPARCCrossCoder,
    classification_df: pd.DataFrame,
    feature_activations: Dict,
    method: str,
) -> Dict:
    compressed_only_features = classification_df[
        classification_df["primary_class"] == "compressed_only"
    ]["feature_id"].tolist()
    
    if len(compressed_only_features) == 0:
        return {
            "superposition_fraction": 0.0,
            "total_compressed_only": 0,
            "superposition_count": 0,
            "features": {},
        }
    
    decoder_weights = crosscoder.get_decoder_weights()
    W_u_dec = decoder_weights["W_u_dec"]
    W_c_dec = decoder_weights["W_c_dec"]
    
    z_u = feature_activations["z_u"]
    z_c = feature_activations["z_c"]
    
    results = {}
    superposition_count = 0
    
    for feature_id in tqdm(compressed_only_features, desc="Analyzing superposition"):
        analysis = analyze_superposition_for_feature(
            feature_id, W_c_dec, W_u_dec, z_c, z_u
        )
        results[feature_id] = analysis
        if analysis["is_superposition"]:
            superposition_count += 1
    
    superposition_fraction = superposition_count / len(compressed_only_features)
    
    return {
        "superposition_fraction": superposition_fraction,
        "total_compressed_only": len(compressed_only_features),
        "superposition_count": superposition_count,
        "features": results,
        "method": method,
    }


def get_superposition_summary(superposition_results: Dict) -> pd.DataFrame:
    records = []
    for feature_id, analysis in superposition_results["features"].items():
        records.append({
            "feature_id": feature_id,
            "r2": analysis["r2"],
            "n_nonzero": analysis["n_nonzero"],
            "is_superposition": analysis["is_superposition"],
            "n_constituents": len(analysis["constituent_features"]),
        })
    return pd.DataFrame(records)


def compare_superposition_wanda_vs_awq(
    wanda_results: Dict,
    awq_results: Dict,
) -> Dict:
    return {
        "wanda": {
            "superposition_fraction": wanda_results["superposition_fraction"],
            "total_compressed_only": wanda_results["total_compressed_only"],
            "superposition_count": wanda_results["superposition_count"],
        },
        "awq": {
            "superposition_fraction": awq_results["superposition_fraction"],
            "total_compressed_only": awq_results["total_compressed_only"],
            "superposition_count": awq_results["superposition_count"],
        },
        "hypothesis_h3_supported": (
            wanda_results["superposition_fraction"] > 0.5 and
            wanda_results["superposition_fraction"] > awq_results["superposition_fraction"]
        ),
    }


def save_superposition_results(results: Dict, output_path: str) -> None:
    import json
    
    serializable = {
        "superposition_fraction": float(results["superposition_fraction"]),
        "total_compressed_only": int(results["total_compressed_only"]),
        "superposition_count": int(results["superposition_count"]),
        "method": str(results.get("method", "unknown")),
        "features": {},
    }
    
    for fid, analysis in results["features"].items():
        serializable["features"][str(fid)] = {
            "feature_id": int(analysis["feature_id"]),
            "r2": float(analysis["r2"]),
            "n_nonzero": int(analysis["n_nonzero"]),
            "is_superposition": bool(analysis["is_superposition"]),
            "constituent_features": [
                {"feature_id": int(c["feature_id"]), "weight": float(c["weight"])}
                for c in analysis["constituent_features"]
            ],
            "top_activating_samples": [int(s) for s in analysis["top_activating_samples"]],
        }
    
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_superposition_results(input_path: str) -> Dict:
    import json
    with open(input_path) as f:
        return json.load(f)
