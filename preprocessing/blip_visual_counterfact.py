"""
VQA on the Visual-Counterfact Dataset — BLIP VQA vs TinyLLaVA toggle
Dataset : https://huggingface.co/datasets/mgolov/Visual-Counterfact
Models  : Salesforce/blip-vqa-base   (BlipForQuestionAnswering + BlipProcessor)
          bczhou/tiny-llava-v1-hf    (LlavaForConditionalGeneration + AutoProcessor)

Install : pip install transformers accelerate datasets Pillow torch

────────────────────────────────────────────────────────────────────────────────
Toggle between models by setting MODEL_CHOICE:
    MODEL_CHOICE = "blip"       -> Salesforce/blip-vqa-base
    MODEL_CHOICE = "tinyllava"  -> bczhou/tiny-llava-v1-hf
────────────────────────────────────────────────────────────────────────────────

FIX NOTES (tiny-llava + newer transformers >= 4.45):
  Error: "Image features and image tokens do not match, tokens: 575, features: 1179648"

  Root cause: newer modeling_llava.py introduced get_placeholder_mask() which runs:
      num_image_features = image_features.numel()   # WRONG
  but image_features has shape [num_images, num_patches, hidden_size], so numel()
  = 1 × 576 × 2048 = 1,179,648 instead of 576.  That is a bug in transformers
  itself for this older checkpoint.

  Additionally, "default" strategy drops the CLS token -> processor inserts 575
  <image> tokens in input_ids, but the vision encoder always outputs 576 patch
  embeddings (24×24).  Switching to "full" keeps CLS so both sides equal 576.

  Fixes applied:
    1. use_fast=False           -> slow CLIPImageProcessor; same layout as training
    2. vision_feature_select_strategy = "full"
                                -> 576 image tokens to match 576 vision features
    3. Monkey-patch get_placeholder_mask
                                -> count features along sequence dim, not numel()
"""

import ast
import types
import torch
from datasets import load_from_disk

# ── Toggle ────────────────────────────────────────────────────────────────────
MODEL_CHOICE = "tinyllava"          # "blip"  |  "tinyllava"

# ── Config ────────────────────────────────────────────────────────────────────
BLIP_MODEL_ID      = "Salesforce/blip-vqa-base"
TINYLLAVA_MODEL_ID = "bczhou/tiny-llava-v1-hf"

COLOR_DATA_DIR = "/data/veer/testing/new/vlm_circuits/data/Visual-Counterfact/color"
SIZE_DATA_DIR  = "/data/veer/testing/new/vlm_circuits/data/Visual-Counterfact/size"
MAX_NEW_TOKENS = 20
BATCH_SIZE     = 8
IMAGE_MODE     = "original"  # "original" | "counterfact" | "both"

# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
print(f"Using device  : {device}")
print(f"Model choice  : {MODEL_CHOICE}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

if MODEL_CHOICE == "blip":
    from transformers import BlipProcessor, BlipForQuestionAnswering

    print(f"Loading {BLIP_MODEL_ID} ...")
    blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
    blip_model = BlipForQuestionAnswering.from_pretrained(
        BLIP_MODEL_ID, torch_dtype=dtype
    ).to(device)
    blip_model.eval()
    print("Model loaded.\n")

elif MODEL_CHOICE == "tinyllava":
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    print(f"Loading {TINYLLAVA_MODEL_ID} ...")

    # FIX 1: use_fast=False forces the slow CLIPImageProcessor.
    # The fast processor (default in newer transformers) outputs differently-shaped
    # tensors that no longer match this checkpoint's expected layout.
    tinyllava_processor = AutoProcessor.from_pretrained(
        TINYLLAVA_MODEL_ID,
        use_fast=False,
    )

    tinyllava_model = LlavaForConditionalGeneration.from_pretrained(
        TINYLLAVA_MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    tinyllava_model.eval()

    # patch_size is missing from the saved processor config — read from model.
    _vcfg = tinyllava_model.config.vision_config
    tinyllava_processor.patch_size = _vcfg.patch_size   # 14

    # FIX 2: switch to "full" so processor inserts 576 <image> tokens (24×24),
    # matching the 576 patch embeddings the vision encoder always produces.
    # "default" drops the CLS token and gives 575 tokens, which is one short.
    tinyllava_processor.vision_feature_select_strategy = "full"
    tinyllava_model.config.vision_feature_select_strategy = "full"

    print(
        f"  patch_size={tinyllava_processor.patch_size}  "
        f"vision_feature_select_strategy="
        f"{tinyllava_processor.vision_feature_select_strategy}"
    )

    # FIX 3: Monkey-patch get_placeholder_mask for transformers >= 4.45.
    # The new code does  num_image_features = image_features.numel()  which for a
    # tensor of shape [1, 576, 2048] gives 1,179,648 instead of 576.
    # We replace the method with one that counts along the sequence dimension only.
    if hasattr(tinyllava_model.model, "get_placeholder_mask"):
        print("  Applying get_placeholder_mask patch for newer transformers ...")

        def _patched_get_placeholder_mask(self, input_ids, image_features, inputs_embeds=None, **kwargs):
            # masked_scatter (called right after this in modeling_llava.py) requires
            # the mask to have the same rank as inputs_embeds, i.e. [B, S, hidden].
            # Returning a 2D [B, S] mask causes the dim-2 size mismatch error.
            # We also skip the token-count validation: this checkpoint has a harmless
            # ±1 CLS-token off-by-one vs current transformers that would always fire.
            image_token_id = self.config.image_token_index
            mask = (input_ids == image_token_id)              # [B, S]
            if inputs_embeds is not None:
                mask = mask.unsqueeze(-1).expand_as(inputs_embeds)  # [B, S, H]
            return mask

        tinyllava_model.model.get_placeholder_mask = types.MethodType(
            _patched_get_placeholder_mask, tinyllava_model.model
        )
        print("  Patch applied.")

    print("Model loaded.\n")

else:
    raise ValueError(f"Unknown MODEL_CHOICE='{MODEL_CHOICE}'. Use 'blip' or 'tinyllava'.")


# ══════════════════════════════════════════════════════════════════════════════
#  QUESTION BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_color_question(obj: str) -> str:
    if MODEL_CHOICE == "blip":
        return f"What color is the {obj}?"
    else:
        return f"USER: <image>\nWhat color is the {obj}?\nASSISTANT:"

def build_size_question(obj: str) -> str:
    if MODEL_CHOICE == "blip":
        return "Which object appears larger in the image?"
    else:
        return "USER: <image>\nWhich object appears larger in the image?\nASSISTANT:"


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_single_blip(image, question: str) -> str:
    inputs = blip_processor(
        images=image,
        text=question,
        return_tensors="pt",
    ).to(device, dtype)
    with torch.no_grad():
        ids = blip_model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    return blip_processor.decode(ids[0], skip_special_tokens=True).strip()


def run_single_tinyllava(image, question: str) -> str:
    """
    Direct model.generate() call — no pipeline.
    Slice prompt tokens from the output before decoding so we only get the
    newly generated answer tokens, not the echoed input.
    """
    inputs = tinyllava_processor(
        text=question,
        images=image,
        return_tensors="pt",
    ).to(device, dtype)

    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated_ids = tinyllava_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # generated_ids contains [prompt_tokens + new_tokens]; keep only new_tokens
    new_ids = generated_ids[:, prompt_len:]
    return tinyllava_processor.tokenizer.decode(
        new_ids[0], skip_special_tokens=True
    ).strip()


def run_batch(images: list, questions: list) -> list:
    runner = run_single_blip if MODEL_CHOICE == "blip" else run_single_tinyllava
    return [runner(img, q) for img, q in zip(images, questions)]


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADING + GOLD ANSWER PARSING
# ══════════════════════════════════════════════════════════════════════════════

print("Loading datasets from disk ...")
color_ds = load_from_disk(COLOR_DATA_DIR)
size_ds  = load_from_disk(SIZE_DATA_DIR)
print(f"  color split : {len(color_ds)} rows  |  columns: {color_ds.column_names}")
print(f"  size  split : {len(size_ds)} rows   |  columns: {size_ds.column_names}\n")


def parse_correct_answer(raw) -> list:
    if isinstance(raw, list):
        return [str(x).lower() for x in raw]
    s = str(raw).strip()
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            return [str(x).lower() for x in parsed]
        except (ValueError, SyntaxError):
            pass
    return [s.lower()]


def is_correct(prediction: str, gold_answers: list) -> bool:
    pred = prediction.lower().strip()
    if not pred:
        return False
    return any(gold in pred for gold in gold_answers)


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_split(dataset, question_fn, split_name: str, image_mode: str = IMAGE_MODE):
    image_keys = []
    if image_mode in ("original", "both"):
        image_keys.append("original_image")
    if image_mode in ("counterfact", "both"):
        image_keys.append("counterfact_image")

    results = {key: [] for key in image_keys}

    for i in range(0, len(dataset), BATCH_SIZE):
        batch     = dataset[i : i + BATCH_SIZE]
        objects   = batch["object"]
        raw_golds = batch["correct_answer"]
        questions = [question_fn(obj) for obj in objects]

        for img_key in image_keys:
            images = batch[img_key]
            preds  = run_batch(images, questions)

            for obj, pred, raw_gold in zip(objects, preds, raw_golds):
                gold_list = parse_correct_answer(raw_gold)
                results[img_key].append({
                    "object":    obj,
                    "question":  question_fn(obj),
                    "predicted": pred,
                    "gold":      gold_list,
                    "correct":   is_correct(pred, gold_list),
                })

        if (i // BATCH_SIZE) % 10 == 0:
            done = min(i + BATCH_SIZE, len(dataset))
            print(f"  [{split_name}] {done}/{len(dataset)} samples processed ...")

    for img_key, rows in results.items():
        n_correct = sum(r["correct"] for r in rows)
        n_empty   = sum(1 for r in rows if not r["predicted"])
        accuracy  = n_correct / len(rows) if rows else 0
        label     = "Original" if img_key == "original_image" else "Counterfactual"

        print(f"\n  [{split_name} | {label}]  Accuracy: {accuracy:.2%}  "
              f"({n_correct}/{len(rows)})  |  Empty predictions: {n_empty}")

        print("  Sample predictions:")
        for r in rows[:5]:
            mark         = "✓" if r["correct"] else "✗"
            pred_display = f'"{r["predicted"]}"' if r["predicted"] else "<empty>"
            print(f"    [{mark}] {r['object']}")
            print(f"         Gold : {r['gold']}")
            print(f"         Pred : {pred_display}")

    return results


# ── Single-sample demo ────────────────────────────────────────────────────────
print("── Single-sample demo (color split, original image) ──")
sample    = color_ds[0]
obj       = sample["object"]
question  = build_color_question(obj)
pred      = run_batch([sample["original_image"]], [question])[0]
gold_list = parse_correct_answer(sample["correct_answer"])
print(f"  Object  : {obj}")
print(f"  Q       : {question}")
print(f"  Gold    : {gold_list}")
print(f"  Pred    : \"{pred}\"")
print(f"  Correct : {is_correct(pred, gold_list)}\n")


# ── Full evaluation ───────────────────────────────────────────────────────────
print("── Evaluating COLOR split ──")
color_results = evaluate_split(color_ds, build_color_question, split_name="color")

print("\n── Evaluating SIZE split ──")
size_results = evaluate_split(size_ds, build_size_question, split_name="size")