import json
import random
from pathlib import Path

import numpy as np
import torch

from . import config


def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_results_dir(model: str, method: str, component: str, token_type: str) -> Path:
    dir_name = f"{model}__{method}__{component}__{token_type}"
    results_dir = config.CROSSCODER_RESULTS_DIR / dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_checkpoint_dir(results_dir: Path) -> Path:
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_activations_dir(results_dir: Path) -> Path:
    activations_dir = results_dir / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)
    return activations_dir


def get_features_dir(results_dir: Path) -> Path:
    features_dir = results_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    return features_dir


def get_metrics_dir(results_dir: Path) -> Path:
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir


def get_plots_dir(results_dir: Path) -> Path:
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def save_checkpoint(model, optimizer, epoch: int, metrics: dict, checkpoint_dir: Path, is_final: bool = False):
    filename = "final.pt" if is_final else f"epoch_{epoch}.pt"
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, checkpoint_dir / filename)


def load_checkpoint(checkpoint_path: Path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=get_device())
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["metrics"]


def save_json(data: dict, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_activations(activations: dict, path: Path):
    torch.save(activations, path)


def load_activations(path: Path) -> dict:
    return torch.load(path, map_location=get_device())


def get_compressed_model_path(model: str, method: str, component: str) -> Path:
    comp_label = component.replace("+", "_")
    return config.COMPRESSED_MODELS_DIR / f"{model}__{method}__{comp_label}"


def flush_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def gpu_memory_info() -> str:
    if not torch.cuda.is_available():
        return "No GPU"
    used = torch.cuda.memory_allocated(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{used:.2f}/{total:.2f} GB"


def get_topk_for_token_type(token_type: str, component: str) -> int:
    if component == "P":
        return config.TOPK_PROJECTOR
    if token_type == "cls":
        return config.TOPK_CLS
    return config.TOPK_PATCH


def get_expansion_factor(component: str) -> int:
    if component == "P":
        return config.EXPANSION_FACTOR_PROJECTOR
    return config.EXPANSION_FACTOR_VISION


def get_activation_dim(model: str, component: str) -> int:
    comp_key = "projector" if component == "P" else "vision"
    return config.ACTIVATION_DIM[model][comp_key]
