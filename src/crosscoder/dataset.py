from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import Dataset

from . import config


class VisualCounterfactDataset(Dataset):
    def __init__(self, split: str = "all"):
        self.metadata = pd.read_csv(config.METADATA_CSV)
        if split != "all":
            self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        
        self.ds_dict = load_from_disk(str(config.VISUAL_COUNTERFACT_DIR))
        self.color_ds = self.ds_dict["color"]
        self.size_ds = self.ds_dict["size"]
        
        self._build_index()
    
    def _build_index(self):
        self.sample_to_ds_idx = {}
        for i, row in enumerate(self.color_ds):
            sample_id = f"visual_counterfact_color_{i}"
            self.sample_to_ds_idx[sample_id] = ("color", i)
        for i, row in enumerate(self.size_ds):
            sample_id = f"visual_counterfact_size_{i}"
            self.sample_to_ds_idx[sample_id] = ("size", i)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.metadata.iloc[idx]
        sample_id = row["sample_id"]
        split_name, ds_idx = self.sample_to_ds_idx[sample_id]
        
        if split_name == "color":
            ds_row = self.color_ds[ds_idx]
        else:
            ds_row = self.size_ds[ds_idx]
        
        return {
            "sample_id": sample_id,
            "image_original": ds_row["original_image"],
            "image_counterfact": ds_row["counterfact_image"],
            "question": row["question"],
            "correct_answer": row["correct_answer"],
            "split": row["split"],
            "source_split": split_name,
        }
    
    def get_all_samples(self) -> List[Dict]:
        return [self[i] for i in range(len(self))]


class PairedActivationDataset(Dataset):
    def __init__(self, activations_u: torch.Tensor, activations_c: torch.Tensor, 
                 sample_ids: List[str], image_types: List[str], splits: List[str]):
        self.activations_u = activations_u
        self.activations_c = activations_c
        self.sample_ids = sample_ids
        self.image_types = image_types
        self.splits = splits
    
    def __len__(self) -> int:
        return len(self.activations_u)
    
    def __getitem__(self, idx: int) -> Dict:
        return {
            "activations_u": self.activations_u[idx],
            "activations_c": self.activations_c[idx],
            "sample_id": self.sample_ids[idx],
            "image_type": self.image_types[idx],
            "split": self.splits[idx],
        }


def create_paired_activation_dataset(
    activations_data: Dict,
    split: str = "train"
) -> PairedActivationDataset:
    mask = [s == split for s in activations_data["splits"]]
    indices = [i for i, m in enumerate(mask) if m]
    
    return PairedActivationDataset(
        activations_u=activations_data["activations_u"][indices],
        activations_c=activations_data["activations_c"][indices],
        sample_ids=[activations_data["sample_ids"][i] for i in indices],
        image_types=[activations_data["image_types"][i] for i in indices],
        splits=[activations_data["splits"][i] for i in indices],
    )


def get_paired_indices(dataset: VisualCounterfactDataset) -> List[Tuple[int, int]]:
    sample_to_idx = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        sample_to_idx[sample["sample_id"]] = i
    
    pairs = []
    seen = set()
    for i in range(len(dataset)):
        sample = dataset[i]
        sample_id = sample["sample_id"]
        if sample_id not in seen:
            pairs.append((i, i))
            seen.add(sample_id)
    
    return pairs


def collate_activations(batch: List[Dict]) -> Dict:
    return {
        "activations_u": torch.stack([b["activations_u"] for b in batch]),
        "activations_c": torch.stack([b["activations_c"] for b in batch]),
        "sample_ids": [b["sample_id"] for b in batch],
        "image_types": [b["image_type"] for b in batch],
        "splits": [b["split"] for b in batch],
    }
