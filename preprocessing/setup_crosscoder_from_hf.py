"""
Self-contained script to load the Visual-Counterfact dataset and compressed VLM
models from Hugging Face, then save them where the crosscoder code expects.

Run from repo root:
  python preprocessing/setup_crosscoder_from_hf.py
  python preprocessing/setup_crosscoder_from_hf.py --dataset-only
  python preprocessing/setup_crosscoder_from_hf.py --models-only

Output:
  - Dataset: <repo_root>/output/counterfactual_selected/
  - Compressed models: <repo_root>/src/compressed_models/ (blip2__wanda__V, etc.)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value, load_dataset

# Paths
_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent
OUTPUT_DIR = REPO_ROOT / "output"
COUNTERFACT_SELECTED_DIR = OUTPUT_DIR / "counterfactual_selected"
SRC_DIR = REPO_ROOT / "src"
COMPRESSED_MODELS_DIR = SRC_DIR / "compressed_models"

# HuggingFace dataset (Visual-Counterfact: color/size splits, VQA)
HF_DATASET_ID = "mgolov/Visual-Counterfact"
# HuggingFace repo for compressed models (subdirs: blip2__wanda__V, blip2__awq__P, ...)
HF_COMPRESSED_MODELS_REPO = "vlm_circuits/compressed_models"

# Train/val split for crosscoder
VAL_FRACTION = 0.2
RANDOM_SEED = 42


def _download_compressed_models(repo_id: str, local_dir: Path) -> None:
    """Download compressed model checkpoints from HF into src/compressed_models/."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("Install huggingface_hub: pip install huggingface-hub") from None

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading compressed models from {repo_id} to {local_dir} ...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        print(f"Done. Compressed models are at: {local_dir}")
    except Exception as e:
        err_msg = str(e).lower()
        if "404" in err_msg or "not found" in err_msg or "repository" in err_msg:
            print(f"Compressed models repo not found or not accessible: {repo_id}")
            print(f"  URL: https://huggingface.co/{repo_id}")
            print("  Upload subdirs (blip2__wanda__V, blip2__awq__P, etc.) or use --dataset-only.")
        raise


def _question_for_split(split_name: str, row: dict) -> str:
    """Generate VQA question per Visual-Counterfact split."""
    obj = row.get("object", "object")
    if split_name == "color":
        return f"What color is the {obj}?"
    return "Which object appears larger in the image?"


def _records_from_hf_ds(hf_ds, split_name: str):
    """Convert one HF split (color or size) to list of records with crosscoder columns."""
    records = []
    for i, row in enumerate(hf_ds):
        # HF dataset uses original_image / counterfact_image (PIL when loaded)
        image_original = row.get("original_image")
        image_counterfact = row.get("counterfact_image")
        if image_original is None or image_counterfact is None:
            continue
        question = _question_for_split(split_name, row)
        records.append({
            "sample_id": f"visual_counterfact_{split_name}_{i}",
            "image_original": image_original,
            "image_counterfact": image_counterfact,
            "question": question,
            "correct_answer": row.get("correct_answer", ""),
            "source_split": split_name,
        })
    return records


def _build_dataset_dict(all_records: list[dict], val_fraction: float, seed: int) -> DatasetDict:
    """Split records into train/val and build DatasetDict with crosscoder split names."""
    rng = random.Random(seed)
    indices = list(range(len(all_records)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_fraction))
    n_train = len(indices) - n_val

    train_records = [all_records[i] for i in indices[:n_train]]
    val_records = [all_records[i] for i in indices[n_train:]]

    # Add "split" column
    for r in train_records:
        r["split"] = "train"
    for r in val_records:
        r["split"] = "val"

    features = Features({
        "sample_id": Value("string"),
        "image_original": HFImage(),
        "image_counterfact": HFImage(),
        "question": Value("string"),
        "correct_answer": Value("string"),
        "split": Value("string"),
        "source_split": Value("string"),
    })

    train_ds = Dataset.from_list(train_records, features=features)
    val_ds = Dataset.from_list(val_records, features=features)

    return DatasetDict({
        "attribute_binding_train": train_ds,
        "attribute_binding_val": val_ds,
    })


def _run_dataset_setup() -> None:
    """Download Visual-Counterfact from HF and save to output/counterfactual_selected."""
    print(f"Loading dataset: {HF_DATASET_ID}")
    ds_dict = load_dataset(HF_DATASET_ID)

    all_records = []
    for split_name in ("color", "size"):
        if split_name not in ds_dict:
            print(f"  Skipping missing split: {split_name}")
            continue
        part = _records_from_hf_ds(ds_dict[split_name], split_name)
        print(f"  {split_name}: {len(part)} samples")
        all_records.extend(part)

    if not all_records:
        raise RuntimeError("No records produced from HF dataset.")

    print(f"\nTotal samples: {len(all_records)}")
    print("Building train/val splits for crosscoder ...")
    out_ds = _build_dataset_dict(all_records, val_fraction=VAL_FRACTION, seed=RANDOM_SEED)

    COUNTERFACT_SELECTED_DIR.parent.mkdir(parents=True, exist_ok=True)
    out_path = str(COUNTERFACT_SELECTED_DIR)
    print(f"Saving to {out_path} ...")
    out_ds.save_to_disk(out_path)
    print("Done. Dataset is at:", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Visual-Counterfact dataset and compressed models from HF for crosscoder."
    )
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Only download and prepare the dataset; skip compressed models.",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Only download compressed models; skip the dataset.",
    )
    parser.add_argument(
        "--models-repo",
        type=str,
        default=HF_COMPRESSED_MODELS_REPO,
        help=f"HuggingFace repo ID for compressed models (default: {HF_COMPRESSED_MODELS_REPO}).",
    )
    args = parser.parse_args()

    do_dataset = not args.models_only
    do_models = not args.dataset_only

    if do_dataset:
        _run_dataset_setup()
    if do_models:
        _download_compressed_models(args.models_repo, COMPRESSED_MODELS_DIR)

    if do_dataset and do_models:
        print("\nCrosscoder expects: output/counterfactual_selected and src/compressed_models/")
    elif do_dataset:
        print("\nCrosscoder dataset path: output/counterfactual_selected (relative to repo root).")
    elif do_models:
        print("\nCompressed models path: src/compressed_models/ (relative to repo root).")
    print("Base models (BLIP-VQA, TinyLLaVA) are loaded by crosscoder from HF by ID (cached by transformers).")


if __name__ == "__main__":
    main()
