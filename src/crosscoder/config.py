from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
SRC_DIR = PROJECT_ROOT / "src"
COMPRESSED_MODELS_DIR = SRC_DIR / "compressed_models"
CROSSCODER_RESULTS_DIR = SRC_DIR / "crosscoder" / "results"

VISUAL_COUNTERFACT_DIR = DATA_DIR / "Visual-Counterfact"
METADATA_CSV = OUTPUT_DIR / "counterfactual_selected_metadata.csv"

SEED = 42
LEARNING_RATE = 3e-4
WARMUP_FRACTION = 0.05
BATCH_SIZE = 32
EXTRACT_BATCH_SIZE = 128
NUM_EPOCHS = 150
CHECKPOINT_EVERY = 100

LAMBDA_SPARSITY = 3e-4
LAMBDA_CROSS = 0.4
GRAD_CLIP_NORM = 1.0
WEIGHT_DECAY = 1e-5
LAMBDA_SHARED_MULTIPLIER = 0.1
FORCED_SHARED_FRACTION = 0.06

TOPK_CLS = 128
TOPK_PATCH = 128
TOPK_PROJECTOR = 128

EXPANSION_FACTOR_VISION = 8 # total neurons in SAE is 768 * 8 = 6144
EXPANSION_FACTOR_PROJECTOR = 16 # total neurons in SAE  768 * 16 = 12288

FVE_THRESHOLD = 0.7
DEAD_NEURON_THRESHOLD = 0.9

RHO_UNCOMPRESSED_ONLY = 0.15
RHO_COMPRESSED_ONLY = 0.85
RHO_SHARED_LOW = 0.35
RHO_SHARED_HIGH = 0.65
THETA_ALIGNED = 0.80
THETA_REDIRECTED = 0.50

SUPERPOSITION_R2_THRESHOLD = 0.8
SUPERPOSITION_MAX_CONSTITUENTS = 3
SUPERPOSITION_TOP_SAMPLES = 100

AWQ_CALIBRATION_SAMPLES = 256

BLIP_VQA_MODEL_ID = "Salesforce/blip-vqa-base"
TINYLLAVA_MODEL_ID = "bczhou/tiny-llava-v1-hf"

MODULE_MAP = {
    "blip2": {
        "vision": "vision_model",
        "projector": "text_encoder",
    },
    "tinyllava": {
        "vision": "model.vision_tower",
        "projector": "model.multi_modal_projector",
    },
}

VISION_HOOK_PATHS = {
    "blip2": "vision_model.encoder.layers",
    "tinyllava": "model.vision_tower.vision_model.encoder.layers",
}

PROJECTOR_HOOK_PATHS = {
    "blip2": "text_encoder.encoder.layer",
    "tinyllava": "model.multi_modal_projector",
}

BLIP_CROSS_ATTENTION_LAYERS = [9, 10, 11]

ACTIVATION_DIM = {
    "blip2": {"vision": 768, "projector": 768},
    "tinyllava": {"vision": 768, "projector": 2048},
}

MODELS = ["blip2", "tinyllava"]
METHODS = ["wanda", "awq"]
COMPONENTS = ["V", "P", "V_P"]
TOKEN_TYPES = ["cls", "patch"]
