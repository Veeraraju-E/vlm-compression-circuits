# Preprocessing: Unified Counterfactual Dataset (Section 3.1)

Build a single HuggingFace dataset from **Visual-Counterfact**, **COCO-Counterfactuals**, and **XAITK Counterfactual VQA**, with quality filtering and train/val splits per circuit type.

## Requirements

- Data in `data/`:
  - `data/Visual-Counterfact/` (from HuggingFace `mgolov/Visual-Counterfact`, on disk)
  - `data/COCO-Counterfactuals/` (from HuggingFace `geoskyr/COCO-Counterfactual`, on disk)
  - `data/XAITK/` with `annotations.json`, `original/*.png`, `counterfactual/*.png`
- Python deps: `torch`, `transformers`, `datasets`, `pandas`, `PIL`

## Pipeline

1. **Load** sources from `data/` (`load_sources.py`).
2. **Run inference** with Qwen3-VL-2B and BLIP-2 on original (and counterfactual) images to get predictions and confidence scores (`run_inference.py`).
3. **Filter** to samples where the base model is correct on the original with confidence > 80% (`filter_and_combine.py`).
4. **Split** by circuit type and create train/val per type (80/20).
5. **Build** a HuggingFace `DatasetDict` with splits like `attribute_binding_train`, `attribute_binding_val`, etc., and export a **single CSV** with the full sample list and both models’ confidence scores (`build_hf_dataset.py`).
6. Optionally **push** the dataset to the HuggingFace Hub.

## Circuit types

- **attribute_binding**: Visual-Counterfact (color + size).
- **object_recognition**: XAITK.
- **answer_generation**: COCO-Counterfactuals (image captioning).

## Run

From the **project root**:

```bash
# Install deps if needed
pip install torch transformers datasets pandas pillow

# Run full pipeline (load → inference → filter → split → build → save CSV and dataset)
cd preprocessing && python build_hf_dataset.py

# Skip inference (e.g. you already ran it and have scores elsewhere)
python build_hf_dataset.py --no-inference

# Require only one model to pass the filter (default: both must pass)
python build_hf_dataset.py --no-require-both

# Export CSV to a custom path
python build_hf_dataset.py --csv output/my_metadata.csv

# Push to HuggingFace Hub (requires login: huggingface-cli login)
python build_hf_dataset.py --push-to-hub
```

Outputs:

- **Dataset**: `output/counterfactual_unified/` (saved with `datasets.save_to_disk`).
- **CSV**: `output/counterfactual_unified_metadata.csv` with columns including `sample_id`, `source`, `circuit_type`, `split`, `qwen3vl_confidence_original`, `blip_confidence_original`, and both models’ predictions.

Set `HF_DATASET_ID` in `config.py` to your Hub repo before `--push-to-hub`.
