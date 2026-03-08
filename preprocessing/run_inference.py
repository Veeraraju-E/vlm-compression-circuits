"""
Run TinyLLaVA and BLIP-2 on counterfactual samples; return predictions and confidence scores.
Confidence = geometric mean of per-token probabilities (from generation scores when available).
"""
from pathlib import Path

import importlib
import types
import torch
from tqdm import tqdm
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoConfig,
    BlipForQuestionAnswering,
    BlipProcessor,
    Blip2ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlavaForConditionalGeneration,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import is_flash_attn_2_available

from config import (
    BLIP2_MODEL_ID,
    BLIP_VQA_MODEL_ID,
    TINYLLAVA_MODEL_ID,
    TINYLLAVA_V1_MODEL_ID,
)

def _patch_tinyllava_tie_weights(model_id: str) -> None:
    """
    TinyLLaVA uses remote code. Some checkpoints override `tie_weights()` with an older
    signature, but recent `transformers` calls:

        model.tie_weights(missing_keys=..., recompute_mapping=False)

    during `from_pretrained()`. Patch the remote class method to accept arbitrary kwargs.
    """
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    auto_map = getattr(cfg, "auto_map", None) or {}
    class_ref = auto_map.get("AutoModelForCausalLM")
    if not class_ref:
        return

    model_cls = get_class_from_dynamic_module(class_ref, model_id)
    if getattr(model_cls, "_vlm_circuits_tie_weights_patched", False):
        return

    orig_tie_weights = getattr(model_cls, "tie_weights", None)
    if orig_tie_weights is None:
        return

    def tie_weights_compat(self, *args, **kwargs):  # noqa: ARG001
        return orig_tie_weights(self)

    model_cls.tie_weights = tie_weights_compat
    model_cls._vlm_circuits_tie_weights_patched = True


def _normalize_answer(s):
    """Normalize for VQA comparison (lowercase, strip, collapse spaces)."""
    if s is None:
        return ""
    return " ".join(str(s).lower().strip().split())


def _check_answer_match(prediction, correct_answer):
    """Check if prediction matches the correct answer.
    
    Handles both single answers and list answers (e.g., ['green', 'blue']).
    For list answers, returns True if prediction contains ANY of the correct answers.
    """
    if correct_answer is None:
        return None
    
    pred_normalized = _normalize_answer(prediction)
    if not pred_normalized:
        return False
    
    # Handle list-format answers (e.g., ['green'] or ['brown', 'white'])
    if isinstance(correct_answer, (list, tuple)):
        for ans in correct_answer:
            ans_normalized = _normalize_answer(ans)
            if ans_normalized and ans_normalized in pred_normalized:
                return True
        return False
    
    # Handle normalized answer format from load_sources
    # (which may be a list of lowercase strings)
    correct_normalized = _normalize_answer(correct_answer)
    
    # Exact match or contained match
    if correct_normalized == pred_normalized:
        return True
    if correct_normalized and correct_normalized in pred_normalized:
        return True
    
    return False


def _caption_match(pred, ref, min_token_overlap=0.3):
    """True if pred is reasonably similar to ref (word overlap ratio)."""
    if not ref or not pred:
        return False
    pred_t = set(_normalize_answer(pred).split())
    ref_t = set(_normalize_answer(ref).split())
    if not ref_t:
        return True
    overlap = len(pred_t & ref_t) / len(ref_t)
    return overlap >= min_token_overlap


def _image_from_record(record, key_original=True):
    """Get PIL Image from record: either image_original/image_counterfact or load from path."""
    if key_original:
        img = record.get("image_original")
        path = record.get("image_original_path")
    else:
        img = record.get("image_counterfact")
        path = record.get("image_counterfact_path")
    if img is not None:
        if isinstance(img, Image.Image):
            # Some sources may include grayscale (mode "L") images; force 3-channel for SigLIP/BLIP.
            return img.convert("RGB")
        return Image.open(img).convert("RGB")
    if path:
        return Image.open(path).convert("RGB")
    return None


def _confidence_from_scores(scores, generated_token_ids):
    """
    Compute geometric mean of per-token probabilities from generation scores.
    scores: tuple of (batch=1, vocab_size) tensors; generated_token_ids: list of int (one per step).
    """
    if not scores or len(scores) == 0 or not generated_token_ids:
        return 0.0
    import torch.nn.functional as F
    log_probs = []
    for logits, token_id in zip(scores, generated_token_ids):
        # `generate(output_scores=True)` returns per-step scores that are usually shaped
        # (batch, vocab), but some configs/models return (vocab,) for batch=1.
        if logits.dim() == 2:
            lp = F.log_softmax(logits[0].float(), dim=-1)  # (vocab,)
        elif logits.dim() == 1:
            lp = F.log_softmax(logits.float(), dim=-1)  # (vocab,)
        else:
            lp = F.log_softmax(logits.reshape(-1).float(), dim=-1)  # best-effort
        log_probs.append(lp[token_id].item())
    if not log_probs:
        return 0.0
    mean_log_p = sum(log_probs) / len(log_probs)
    return float(torch.exp(torch.tensor(mean_log_p)).item())


def _get_attn_implementation(prefer_flash_attention: bool) -> str:
    """
    Return the attention implementation string for transformers `from_pretrained`.

    - "flash_attention_2" requires the `flash-attn` package and a compatible CUDA build.
    - TinyLLaVA remote code is not fully compatible with transformers attention capability checks,
      so we keep it on eager regardless.
    """
    if prefer_flash_attention and is_flash_attn_2_available():
        return "flash_attention_2"
    return "eager"


class BlipVQAInference:
    """
    Visual-Counterfact is a VQA task; use a VQA-trained BLIP checkpoint.

    This mirrors `preprocessing/blip_visual_counterfact.py`:
    `BlipProcessor` + `BlipForQuestionAnswering`.
    """

    def __init__(self, model_id=BLIP_VQA_MODEL_ID, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForQuestionAnswering.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)
        self.model.eval()

    def run_vqa(self, image, question, max_new_tokens=20):
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device, self.dtype)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        num_steps = len(out.scores) if out.scores else 0
        if num_steps == 0:
            return "", 0.0
        gen_ids = out.sequences[0, -num_steps:]
        text = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        conf = _confidence_from_scores(out.scores, gen_ids.tolist())
        return text, conf

    def run_vqa_batch(self, images, questions, max_new_tokens=20):
        # Mirror the standalone script behavior: run per-sample (no true batching).
        preds = []
        confs = []
        for img, q in zip(images, questions):
            p, c = self.run_vqa(img, q, max_new_tokens=max_new_tokens)
            preds.append(p)
            confs.append(c)
        return preds, confs


class TinyLLaVAV1Inference:
    """
    TinyLLaVA inference for Visual-Counterfact matching `preprocessing/blip_visual_counterfact.py`.

    - Model: `bczhou/tiny-llava-v1-hf` via `LlavaForConditionalGeneration`
    - Processor: `AutoProcessor(use_fast=False)`
    - Prompt format:
        USER: <image>
        {question}
        ASSISTANT:
    """

    def __init__(self, model_id=TINYLLAVA_V1_MODEL_ID, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        vcfg = self.model.config.vision_config
        self.processor.patch_size = vcfg.patch_size
        self.processor.vision_feature_select_strategy = "full"
        self.model.config.vision_feature_select_strategy = "full"

        if hasattr(self.model.model, "get_placeholder_mask"):
            def _patched_get_placeholder_mask(self_, input_ids, image_features, inputs_embeds=None, **kwargs):
                image_token_id = self_.config.image_token_index
                mask = (input_ids == image_token_id)  # [B, S]
                if inputs_embeds is not None:
                    mask = mask.unsqueeze(-1).expand_as(inputs_embeds)  # [B, S, H]
                return mask

            self.model.model.get_placeholder_mask = types.MethodType(_patched_get_placeholder_mask, self.model.model)

    @staticmethod
    def _format_prompt(question: str) -> str:
        return f"USER: <image>\\n{question}\\nASSISTANT:"

    def run_vqa(self, image, question, max_new_tokens=20):
        preds, confs = self.run_vqa_batch([image], [question], max_new_tokens=max_new_tokens)
        return preds[0], confs[0]

    def run_vqa_batch(self, images, questions, max_new_tokens=20):
        # Mirror `preprocessing/blip_visual_counterfact.py` exactly:
        # process each sample individually, slice off prompt tokens using prompt_len,
        # then decode only the newly generated tokens.
        preds = []
        confs = []
        for img, q in zip(images, questions):
            prompt = self._format_prompt(q)
            inputs = self.processor(
                text=prompt,
                images=img,
                return_tensors="pt",
            ).to(self.device, self.dtype)

            prompt_len = inputs["input_ids"].shape[1]

            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

            # generated_ids contains [prompt_tokens + new_tokens]; keep only new_tokens
            new_ids = out.sequences[:, prompt_len:][0]
            text = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            preds.append(text)

            if out.scores and new_ids.numel() > 0:
                k = min(len(out.scores), new_ids.numel())
                confs.append(_confidence_from_scores(out.scores[:k], new_ids[:k].tolist()))
            else:
                confs.append(0.0)

        return preds, confs


class Blip2Inference:
    """BLIP-2 OPT-2.7B inference for VQA and captioning.
    
    For VQA: Uses the format "Question: {q} Short answer:" which produces concise answers.
    For captioning: Uses unconditional generation.
    """
    
    def __init__(self, model_id=BLIP2_MODEL_ID, device=None, prefer_flash_attention=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else None,
            attn_implementation=_get_attn_implementation(prefer_flash_attention),
        ).to(self.device)
        self.model.eval()

    def _format_vqa_prompt(self, question):
        """Format VQA prompt for BLIP-2 to produce short answers.
        
        BLIP-2 works best with "Question: X Short answer:" format for concise VQA responses.
        """
        return f"Question: {question} Short answer:"

    def run_vqa(self, image, question, max_new_tokens=10):
        """VQA: prompt with question, return (answer_text, confidence).
        
        Uses shorter max_new_tokens (10) for VQA to encourage concise answers.
        """
        prompt = self._format_vqa_prompt(question)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            num_beams=1,
        )
        seq = out.sequences[0]
        pred_ids = seq[inputs["input_ids"].shape[1]:]
        answer = self.processor.decode(pred_ids, skip_special_tokens=True).strip()
        conf = 0.0
        if out.scores and pred_ids.numel() > 0:
            conf = _confidence_from_scores(out.scores, pred_ids.tolist())
        return answer, conf

    def run_caption(self, image, max_new_tokens=40):
        """Caption: unconditional generation, return (caption_text, confidence)."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            num_beams=1,
        )
        seq = out.sequences[0]
        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        pred_ids = seq[input_len:]
        caption = self.processor.decode(pred_ids, skip_special_tokens=True).strip()
        conf = 0.0
        if out.scores and pred_ids.numel() > 0:
            conf = _confidence_from_scores(out.scores, pred_ids.tolist())
        return caption, conf

    def run_vqa_batch(self, images, questions, max_new_tokens=10):
        """Batch VQA inference. Returns (predictions, confidences)."""
        prompts = [self._format_vqa_prompt(q) for q in questions]
        inputs = self.processor(images=images, text=prompts, return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            num_beams=1,
        )
        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        preds = []
        confs = []
        for i in range(out.sequences.shape[0]):
            pred_ids = out.sequences[i, input_len:]
            preds.append(self.processor.decode(pred_ids, skip_special_tokens=True).strip())
            if out.scores and pred_ids.numel() > 0:
                sample_scores = [s[i:i+1] for s in out.scores]
                confs.append(_confidence_from_scores(sample_scores, pred_ids.tolist()))
            else:
                confs.append(0.0)
        return preds, confs

    def run_caption_batch(self, images, max_new_tokens=40):
        """Batch captioning inference. Returns (predictions, confidences)."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            num_beams=1,
        )
        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        preds = []
        confs = []
        for i in range(out.sequences.shape[0]):
            pred_ids = out.sequences[i, input_len:]
            preds.append(self.processor.decode(pred_ids, skip_special_tokens=True).strip())
            if out.scores and pred_ids.numel() > 0:
                sample_scores = [s[i:i+1] for s in out.scores]
                confs.append(_confidence_from_scores(sample_scores, pred_ids.tolist()))
            else:
                confs.append(0.0)
        return preds, confs


class TinyLLaVAInference:
    """TinyLLaVA-3.1B (Phi-2 + SigLIP) inference for VQA and captioning.
    
    Uses the conversation template format from TinyLLaVA's remote code:
    - System prompt + "USER: <image>\\n{question} ASSISTANT:"
    - The model generates after ASSISTANT:
    """

    def __init__(self, model_id=TINYLLAVA_MODEL_ID, device=None, prefer_flash_attention=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _patch_tinyllava_tie_weights(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="eager",
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
        self._tinyllava_mod = importlib.import_module(self.model.__class__.__module__)

    def run_vqa(self, image, question, max_new_tokens=20):
        """VQA: Answer the question about the image."""
        return self._chat_generate(prompt=question, image=image, max_new_tokens=max_new_tokens)

    def run_caption(self, image, max_new_tokens=40):
        """Caption: Describe the image."""
        return self._chat_generate(
            prompt="Describe this image briefly.",
            image=image,
            max_new_tokens=max_new_tokens,
        )

    def _chat_generate(self, prompt, image, max_new_tokens):
        """Generate response using TinyLLaVA's conversation format.
        
        The output sequence from generate() with inputs_embeds does NOT include the input tokens.
        The returned sequences contain ONLY the generated tokens when using inputs_embeds.
        We decode these directly to get the response text.
        """
        mod = self._tinyllava_mod
        image_processor = self.model.vision_tower._image_processor

        # Build conversation prompt
        formatted_prompt = mod.DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = mod.conv_phi_v0.copy()
        conv.append_message(conv.roles[0], formatted_prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        image = image.convert("RGB")
        image_tensor = mod.process_images([image], image_processor, self.model.config).to(self.device)
        input_ids = mod.tokenizer_image_token(
            full_prompt, self.tokenizer, mod.IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        attention_mask = torch.ones_like(input_ids)

        out = self.model.generate(
            input_ids,
            images=image_tensor,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_token_id,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Extract generated text
        # The model's generate with inputs_embeds returns the full sequence including prompt tokens
        # But after multimodal embedding, the sequence length changes
        # Use the number of scores as the definitive count of generated tokens
        num_generated = len(out.scores) if out.scores else 0
        
        if num_generated > 0:
            # Get the last num_generated tokens from the sequence - these are the generated tokens
            generated_ids = out.sequences[0, -num_generated:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            conf = _confidence_from_scores(out.scores, generated_ids.tolist())
        else:
            text = ""
            conf = 0.0
        
        return text, conf

    def run_vqa_batch(self, images, questions, max_new_tokens=20):
        """Batch VQA - processes sequentially as TinyLLaVA doesn't support true batching."""
        preds = []
        confs = []
        for img, q in zip(images, questions):
            p, c = self._chat_generate(prompt=q, image=img, max_new_tokens=max_new_tokens)
            preds.append(p)
            confs.append(c)
        return preds, confs

    def run_caption_batch(self, images, max_new_tokens=40):
        """Batch captioning - processes sequentially."""
        preds = []
        confs = []
        for img in images:
            p, c = self._chat_generate(prompt="Describe this image briefly.", image=img, max_new_tokens=max_new_tokens)
            preds.append(p)
            confs.append(c)
        return preds, confs


def run_both_models_on_record(record, blip, tinyllava, task="vqa", question=None, correct_answer=None, caption_ref=None):
    """
    Run both models on the original image only (no counterfactual predictions).
    
    For VQA (Visual-Counterfact): Run VQA on original image, compare to correct answer.
    For caption (COCO-Counterfactuals):
        - text_counterfactual: Run caption on single image
        - image_counterfactual: Run caption on both images
    """
    img_orig = _image_from_record(record, key_original=True)
    if img_orig is None:
        return

    counterfactual_type = record.get("counterfactual_type")

    if task == "vqa":
        q = question or record.get("question") or ""
        t_pred_orig, t_conf_orig = tinyllava.run_vqa(img_orig, q)
        b_pred_orig, b_conf_orig = blip.run_vqa(img_orig, q)
        gt = correct_answer or record.get("correct_answer_normalized") or record.get("correct_answer")
        t_correct = _check_answer_match(t_pred_orig, gt)
        b_correct = _check_answer_match(b_pred_orig, gt)
        
        record["tinyllava_pred_original"] = t_pred_orig
        record["tinyllava_confidence_original"] = t_conf_orig
        record["blip_pred_original"] = b_pred_orig
        record["blip_confidence_original"] = b_conf_orig
        record["tinyllava_correct_original"] = t_correct
        record["blip_correct_original"] = b_correct
        
    elif task == "caption":
        t_pred_orig, t_conf_orig = tinyllava.run_caption(img_orig)
        b_pred_orig, b_conf_orig = blip.run_caption(img_orig)
        ref = caption_ref or record.get("caption_original") or record.get("correct_answer")
        t_correct = _caption_match(t_pred_orig, ref)
        b_correct = _caption_match(b_pred_orig, ref)
        
        record["tinyllava_pred_original"] = t_pred_orig
        record["tinyllava_confidence_original"] = t_conf_orig
        record["blip_pred_original"] = b_pred_orig
        record["blip_confidence_original"] = b_conf_orig
        record["tinyllava_correct_original"] = t_correct
        record["blip_correct_original"] = b_correct
        
        # For image counterfactual, also run on the counterfactual image
        if counterfactual_type == "image":
            img_cf = _image_from_record(record, key_original=False)
            if img_cf is not None:
                t_pred_cf, t_conf_cf = tinyllava.run_caption(img_cf)
                b_pred_cf, b_conf_cf = blip.run_caption(img_cf)
                ref_cf = record.get("caption_counterfact")
                record["tinyllava_pred_counterfact"] = t_pred_cf
                record["tinyllava_confidence_counterfact"] = t_conf_cf
                record["blip_pred_counterfact"] = b_pred_cf
                record["blip_confidence_counterfact"] = b_conf_cf
                record["tinyllava_correct_counterfact"] = _caption_match(t_pred_cf, ref_cf)
                record["blip_correct_counterfact"] = _caption_match(b_pred_cf, ref_cf)


def run_inference_on_records(
    records,
    device=None,
    batch_size=2,
    prefer_flash_attention=True,
    save_path=None,
):
    """Run inference and (optionally) append outputs to JSONL.

    Current scope: Visual-Counterfact only (VQA on original image only).
    Mirrors the standalone Visual-Counterfact scripts:
    - BLIP VQA: `Salesforce/blip-vqa-base`
    - TinyLLaVA: `bczhou/tiny-llava-v1-hf`
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    # keep arg for compatibility; not used in this VQA-only path
    _ = prefer_flash_attention

    blip = BlipVQAInference(device=device)
    tinyllava = TinyLLaVAV1Inference(device=device)

    vqa_records = [r for r in records if r.get("task", "vqa") == "vqa"]

    def _append_jsonl(recs):
        if save_path is None:
            return
        import json

        save_path_p = Path(save_path)
        save_path_p.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path_p, "a") as f:
            for rr in recs:
                out = {k: v for k, v in rr.items() if not k.startswith("image_")}
                f.write(json.dumps(out) + "\n")

    # VQA records (Visual-Counterfact): Run on original image only
    for i in tqdm(range(0, len(vqa_records), batch_size), desc="VQA inference", unit="batch"):
        batch = vqa_records[i:i + batch_size]
        images_o = [_image_from_record(r, key_original=True) for r in batch]
        questions = [(r.get("question") or "") for r in batch]

        t_pred_o, t_conf_o = tinyllava.run_vqa_batch(images_o, questions)
        b_pred_o, b_conf_o = blip.run_vqa_batch(images_o, questions)

        for r, tpo, bpo, tconf, bconf in zip(batch, t_pred_o, b_pred_o, t_conf_o, b_conf_o):
            r["tinyllava_pred_original"] = tpo
            r["tinyllava_confidence_original"] = tconf
            r["blip_pred_original"] = bpo
            r["blip_confidence_original"] = bconf
            gt = r.get("correct_answer_normalized") or r.get("correct_answer")
            r["tinyllava_correct_original"] = _check_answer_match(tpo, gt)
            r["blip_correct_original"] = _check_answer_match(bpo, gt)
        _append_jsonl(batch)
