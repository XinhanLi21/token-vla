#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_qwen25vl_vla_grouped_lr_3img.py

Train Qwen2.5-VL-3B (with 2048 traj tokens already added into tokenizer) to predict
FUTURE trajectory tokens, given:
  - 3 images: front_left (left), zed_left (middle), front_right (right)
  - driving prompt text (e.g., "Go straight" / "Turn left" ...)
  - vehicle state (speed/steer/yaw_rate etc. best-effort from your json)
  - past 3s traj tokens (6 tokens @ dt=0.5s)

Output:
  - future 5s traj tokens (10 tokens @ dt=0.5s)

Features:
  - JSONL offset Dataset (memory-light)
  - DataCollator using Qwen chat_template + multimodal processor
  - Grouped learning rates:
      * lr_traj applied to embeddings + lm_head (where traj tokens live)
      * lr_base applied to the rest
  - Optional full fine-tuning (default) or freeze_base mode
  - Adds --processor_dir to avoid HF Hub validation issues

Example:
python train_qwen25vl_vla_grouped_lr_3img.py \
  --model_dir "/home/lxh/lxh/12.26/qwen2_5vl_with_traj_tokens_2048" \
  --processor_dir "/home/lxh/lxh/12.26/qwen2_5vl_with_traj_tokens_2048" \
  --train_jsonl "/home/lxh/traj-vla/12.27-auto-vla/records_ctx_tokens_only_past3s_future5s_dt0p5.jsonl" \
  --base_dir "/home/lxh/lxh/vla_data" \
  --output_dir "/home/lxh/traj-vla/12.27-auto-vla/ckpt_grouped_full" \
  --lr_base 1e-5 --lr_traj 1e-4 \
  --batch_size 1 --grad_accum 32 --epochs 1 \
  --num_workers 4 --enable_gc --require_prompt
"""

import os
import gc
import re
import json
import math
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)


# -------------------------
# Path utils
# -------------------------
def _is_abs(p: str) -> bool:
    try:
        return os.path.isabs(p)
    except Exception:
        return False


def _join_base(base_dir: str, p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    p = str(p)
    if _is_abs(p):
        return p
    return os.path.join(base_dir, p)


def _resolve_dir(p: str) -> str:
    return str(Path(p).expanduser().resolve())


def _has_processor_files(p: str) -> bool:
    return (
        os.path.exists(os.path.join(p, "preprocessor_config.json"))
        or os.path.exists(os.path.join(p, "processor_config.json"))
    )


# -------------------------
# Token helpers
# -------------------------
def find_subsequence(haystack: List[int], needle: List[int]) -> int:
    """return first index of needle in haystack, -1 if not found"""
    if len(needle) == 0:
        return 0
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def build_traj_token_str(token_id: int, prefix: str = "traj") -> str:
    return f"<{prefix}_{int(token_id):04d}>"


def build_traj_token_ids(tokenizer, K: int, prefix: str) -> List[int]:
    toks = [f"<{prefix}_{i:04d}>" for i in range(K)]
    ids = tokenizer.convert_tokens_to_ids(toks)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    bad = []
    for i, tid in enumerate(ids):
        if tid is None or (unk_id is not None and tid == unk_id):
            bad.append((i, toks[i], tid))
    if bad:
        raise ValueError(f"Some traj tokens are missing in tokenizer vocab. examples={bad[:5]}")
    return ids


# -------------------------
# Vehicle state extraction (best-effort)
# -------------------------
def safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def get_speed_mps(rec: Dict[str, Any]) -> Optional[float]:
    # prefer your cached field if present
    if "_speed_mps" in rec:
        v = safe_float(rec.get("_speed_mps"))
        if v is not None:
            return v

    # canfd.spd seems in km/h? (your logs show 15.4125; looks like km/h)
    canfd = rec.get("canfd", {})
    if isinstance(canfd, dict) and "spd" in canfd:
        spd = safe_float(canfd.get("spd"))
        if spd is not None:
            # heuristic: treat as km/h if it's > 4 and typical
            # keep consistent: convert to m/s
            return spd / 3.6

    # imu.807.Vel is m/s in your sample
    imu = rec.get("imu", {})
    if isinstance(imu, dict):
        nav807 = imu.get("807", {})
        if isinstance(nav807, dict) and "Vel" in nav807:
            v = safe_float(nav807.get("Vel"))
            if v is not None:
                return v
    return None


def get_steer_deg(rec: Dict[str, Any]) -> Optional[float]:
    canfd = rec.get("canfd", {})
    if isinstance(canfd, dict) and "steer" in canfd:
        return safe_float(canfd.get("steer"))
    return None


def get_yawrate_radps(rec: Dict[str, Any]) -> Optional[float]:
    # imu.801.AngRateRawZ maybe rad/s? your sample "1.36" looks plausible
    imu = rec.get("imu", {})
    if isinstance(imu, dict):
        nav801 = imu.get("801", {})
        if isinstance(nav801, dict) and "AngRateRawZ" in nav801:
            return safe_float(nav801.get("AngRateRawZ"))
    return None


def get_heading_deg(rec: Dict[str, Any]) -> Optional[float]:
    imu = rec.get("imu", {})
    if isinstance(imu, dict):
        nav810 = imu.get("810", {})
        if isinstance(nav810, dict) and "AngleHeading" in nav810:
            return safe_float(nav810.get("AngleHeading"))
    return None


# -------------------------
# JSONL dataset with offsets (memory-light)
# -------------------------
class JsonlOffsetDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.jsonl_path = str(Path(jsonl_path).expanduser().resolve())
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(self.jsonl_path)

        self.offsets: List[int] = []
        with open(self.jsonl_path, "rb") as f:
            off = 0
            for line in f:
                if line.strip():
                    self.offsets.append(off)
                off += len(line)

        if len(self.offsets) == 0:
            raise RuntimeError(f"Empty jsonl: {self.jsonl_path}")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        off = self.offsets[idx]
        with open(self.jsonl_path, "rb") as f:
            f.seek(off)
            line = f.readline()
        obj = json.loads(line.decode("utf-8"))
        return obj


# -------------------------
# Collator (3 images + past tokens in input, future tokens as assistant)
# -------------------------
@dataclass
class DataCollatorQwenVLPastFutureTokens3Img:
    processor: Any
    tokenizer: Any
    base_dir: str
    token_ctx_key: str
    prefix: str = "traj"
    max_length: int = 2048
    ignore_index: int = -100

    # image keys (can be overridden by CLI)
    img_left_key: str = "front_left"
    img_mid_key: str = "zed_left"
    img_right_key: str = "front_right"

    require_prompt: bool = False

    def _format_prompt(self, rec: Dict[str, Any]) -> str:
        p = rec.get("prompt", "")
        if self.require_prompt and (not isinstance(p, str) or len(p.strip()) == 0):
            raise ValueError("require_prompt=True but record has empty/missing 'prompt'")
        return str(p).strip()

    def _format_vehicle_state(self, rec: Dict[str, Any]) -> str:
        v = get_speed_mps(rec)
        steer = get_steer_deg(rec)
        yawrate = get_yawrate_radps(rec)
        heading = get_heading_deg(rec)

        parts = []
        if v is not None:
            parts.append(f"speed_mps={v:.3f}")
        if steer is not None:
            parts.append(f"steer_deg={steer:.2f}")
        if yawrate is not None:
            parts.append(f"yaw_rate_radps={yawrate:.3f}")
        if heading is not None:
            parts.append(f"heading_deg={heading:.2f}")
        if not parts:
            return "vehicle_state: N/A"
        return "vehicle_state: " + ", ".join(parts)

    def _format_past_tokens(self, rec: Dict[str, Any]) -> str:
        ctx = rec.get(self.token_ctx_key, None)
        if not isinstance(ctx, dict):
            raise KeyError(f"Missing token ctx dict: {self.token_ctx_key}")

        past_ids = ctx.get("tokens_past_3s", None)
        fut_ids = ctx.get("tokens_future_5s", None)
        if not isinstance(past_ids, list) or not isinstance(fut_ids, list):
            raise KeyError(f"token ctx missing tokens_past_3s or tokens_future_5s in {self.token_ctx_key}")

        # expect 6 and 10, but don't hard fail; just use what exists
        past_str = " ".join(build_traj_token_str(int(t), self.prefix) for t in past_ids)
        fut_str = " ".join(build_traj_token_str(int(t), self.prefix) for t in fut_ids)
        return past_str, fut_str

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images_batch: List[List[Image.Image]] = []
        full_texts: List[str] = []
        prompt_texts: List[str] = []

        for ex in batch:
            # images
            paths = ex.get("paths", {})
            if not isinstance(paths, dict):
                raise KeyError("Missing or invalid 'paths' dict")

            p_left = _join_base(self.base_dir, paths.get(self.img_left_key))
            p_mid = _join_base(self.base_dir, paths.get(self.img_mid_key))
            p_right = _join_base(self.base_dir, paths.get(self.img_right_key))

            for p in [p_left, p_mid, p_right]:
                if (p is None) or (not os.path.exists(p)):
                    raise FileNotFoundError(f"Image not found: {p}")

            img_left = Image.open(p_left).convert("RGB")
            img_mid = Image.open(p_mid).convert("RGB")
            img_right = Image.open(p_right).convert("RGB")
            images_batch.append([img_mid, img_left, img_right])  # keep order consistent with template

            driving_prompt = self._format_prompt(ex)
            veh_state = self._format_vehicle_state(ex)
            past_str, fut_str = self._format_past_tokens(ex)

            # User instruction: include prompt ("go straight or turn ..."), state, past tokens; output future tokens.
            user_text = (
                "You are an autonomous driving agent.\n"
                "You are given three camera images: middle, left, right.\n"
                "Task: predict the FUTURE trajectory tokens for the next 5.0 seconds.\n\n"
                f"Driving prompt (high-level intent): {driving_prompt}\n"
                f"{veh_state}\n"
                f"Past 3.0s trajectory tokens (dt=0.5s): {past_str}\n\n"
                "Output ONLY the future tokens as a sequence of <traj_XXXX> tokens (10 tokens for 5.0s at dt=0.5s), separated by spaces.\n"
            )

            mm_msgs_full = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_mid},
                        {"type": "image", "image": img_left},
                        {"type": "image", "image": img_right},
                        {"type": "text", "text": user_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": fut_str}],
                },
            ]

            full_text = self.processor.apply_chat_template(
                mm_msgs_full, tokenize=False, add_generation_prompt=False
            )

            mm_msgs_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_mid},
                        {"type": "image", "image": img_left},
                        {"type": "image", "image": img_right},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
            prompt_text = self.processor.apply_chat_template(
                mm_msgs_prompt, tokenize=False, add_generation_prompt=True
            )

            full_texts.append(full_text)
            prompt_texts.append(prompt_text)

        # tokenize full (with images)
        model_inputs = self.processor(
            text=full_texts,
            images=images_batch,  # list[list[PIL]]
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # tokenize prompt (text only) to know how much to mask
        prompt_enc = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()

        for b in range(input_ids.size(0)):
            full_ids = input_ids[b].tolist()
            prompt_ids = prompt_enc["input_ids"][b].tolist()

            pad_id = self.tokenizer.pad_token_id
            if pad_id is not None:
                while len(prompt_ids) > 0 and prompt_ids[-1] == pad_id:
                    prompt_ids.pop()

            start = find_subsequence(full_ids, prompt_ids)
            if start == -1:
                prompt_len = int(prompt_enc["attention_mask"][b].sum().item())
                labels[b, :prompt_len] = self.ignore_index
            else:
                labels[b, : start + len(prompt_ids)] = self.ignore_index

        model_inputs["labels"] = labels
        return model_inputs


# -------------------------
# Trainer: grouped LR optimizer + optional GC
# -------------------------
class GroupedLRTrainer(Trainer):
    def __init__(self, *args, lr_base: float, lr_traj: float, weight_decay: float,
                 enable_gc: bool = False, gc_interval: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self._lr_base = float(lr_base)
        self._lr_traj = float(lr_traj)
        self._weight_decay = float(weight_decay)
        self._enable_gc = bool(enable_gc)
        self._gc_interval = int(gc_interval)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        wd = self._weight_decay

        # We apply lr_traj to embeddings + lm_head; lr_base to the rest
        in_emb = model.get_input_embeddings()
        out_emb = model.get_output_embeddings()

        traj_param_ids = set()
        if in_emb is not None:
            traj_param_ids.add(id(in_emb.weight))
        if out_emb is not None and hasattr(out_emb, "weight"):
            traj_param_ids.add(id(out_emb.weight))

        base_decay, base_nodecay = [], []
        traj_decay, traj_nodecay = [], []

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_traj = id(p) in traj_param_ids

            # no_decay rule
            no_decay = any(k in n.lower() for k in ["bias", "layernorm", "layer_norm", "ln_", "norm"])
            if is_traj:
                (traj_nodecay if no_decay else traj_decay).append(p)
            else:
                (base_nodecay if no_decay else base_decay).append(p)

        optim_groups = []
        if base_decay:
            optim_groups.append({"params": base_decay, "lr": self._lr_base, "weight_decay": wd})
        if base_nodecay:
            optim_groups.append({"params": base_nodecay, "lr": self._lr_base, "weight_decay": 0.0})
        if traj_decay:
            optim_groups.append({"params": traj_decay, "lr": self._lr_traj, "weight_decay": wd})
        if traj_nodecay:
            optim_groups.append({"params": traj_nodecay, "lr": self._lr_traj, "weight_decay": 0.0})

        if not optim_groups:
            raise RuntimeError("No trainable parameters found. Did you freeze everything?")

        self.optimizer = torch.optim.AdamW(optim_groups, betas=(0.9, 0.999), eps=1e-8)
        return self.optimizer

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)

        if self._enable_gc and (self.state.global_step % self._gc_interval == 0):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return loss


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--processor_dir", type=str, default=None,
                    help="Directory that contains processor/tokenizer files. If None, use --model_dir.")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--prefix", type=str, default="traj")

    ap.add_argument("--token_ctx_key", type=str, default="token_ctx_dt0p5_past3s_future5s",
                    help="Key in each record containing tokens_past_3s and tokens_future_5s")

    ap.add_argument("--img_left_key", type=str, default="front_left")
    ap.add_argument("--img_mid_key", type=str, default="zed_left")
    ap.add_argument("--img_right_key", type=str, default="front_right")

    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)

    ap.add_argument("--lr_base", type=float, default=1e-5)
    ap.add_argument("--lr_traj", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--require_prompt", action="store_true",
                    help="If set, drop/raise on empty prompt (intent).")

    ap.add_argument("--train_mode", type=str, default="full",
                    choices=["full", "freeze_base"],
                    help="full: finetune all params; freeze_base: only embeddings+lm_head trainable (grouped LR still applies).")

    ap.add_argument("--enable_gc", action="store_true")
    ap.add_argument("--gc_interval", type=int, default=50)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_dir = _resolve_dir(args.model_dir)
    proc_dir = _resolve_dir(args.processor_dir) if args.processor_dir else model_dir
    base_dir = _resolve_dir(args.base_dir)
    train_jsonl = str(Path(args.train_jsonl).expanduser().resolve())
    out_dir = _resolve_dir(args.output_dir)

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model_dir not found: {model_dir}")
    if not os.path.isdir(proc_dir):
        raise FileNotFoundError(f"processor_dir not found: {proc_dir}")
    if not _has_processor_files(proc_dir):
        raise RuntimeError(
            f"Processor config missing in {proc_dir}. Need preprocessor_config.json or processor_config.json."
        )
    if not os.path.exists(train_jsonl):
        raise FileNotFoundError(train_jsonl)

    print("[Paths]")
    print("  model_dir     :", model_dir)
    print("  processor_dir :", proc_dir)
    print("  train_jsonl   :", train_jsonl)
    print("  base_dir      :", base_dir)
    print("  output_dir    :", out_dir)

    # ---- processor/tokenizer ----
    processor = AutoProcessor.from_pretrained(
        proc_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = processor.tokenizer

    # ---- model ----
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False

    # ---- ensure embeddings resized (if tokenizer expanded) ----
    if getattr(model.config, "vocab_size", None) != len(tokenizer):
        print(f"[Resize] model vocab_size={getattr(model.config,'vocab_size',None)} -> tokenizer_len={len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    # ---- verify traj tokens exist ----
    _ = build_traj_token_ids(tokenizer, K=args.K, prefix=args.prefix)

    # ---- training mode ----
    if args.train_mode == "freeze_base":
        # train only embeddings + lm_head (so action tokens learn faster, avoid base drifting)
        for p in model.parameters():
            p.requires_grad = False
        in_emb = model.get_input_embeddings()
        out_emb = model.get_output_embeddings()
        if in_emb is None or out_emb is None:
            raise RuntimeError("Model embeddings not found.")
        in_emb.weight.requires_grad = True
        if hasattr(out_emb, "weight"):
            out_emb.weight.requires_grad = True
        print("[Mode] freeze_base: only input_embeddings + lm_head trainable")
    else:
        # full finetune
        for p in model.parameters():
            p.requires_grad = True
        print("[Mode] full: all parameters trainable")

    # ---- dataset & collator ----
    ds = JsonlOffsetDataset(train_jsonl)
    collator = DataCollatorQwenVLPastFutureTokens3Img(
        processor=processor,
        tokenizer=tokenizer,
        base_dir=base_dir,
        token_ctx_key=args.token_ctx_key,
        prefix=args.prefix,
        max_length=args.max_length,
        ignore_index=-100,
        img_left_key=args.img_left_key,
        img_mid_key=args.img_mid_key,
        img_right_key=args.img_right_key,
        require_prompt=args.require_prompt,
    )

    bf16 = bool(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)

    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,

        logging_steps=args.log_every,
        save_steps=args.save_every,
        save_total_limit=2,

        bf16=bf16,
        fp16=False,

        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,  # MUST for multimodal
        report_to="none",

        optim="adamw_torch",          # we override create_optimizer anyway
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,

        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )

    trainer = GroupedLRTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        lr_base=args.lr_base,
        lr_traj=args.lr_traj,
        weight_decay=args.weight_decay,
        enable_gc=args.enable_gc,
        gc_interval=args.gc_interval,
    )

    trainer.train()

    # Save
    trainer.model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print("✅ saved to:", out_dir)


if __name__ == "__main__":
    main()
