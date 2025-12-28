#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train Qwen2.5-VL-3B VLA (3 images + prompt + state + past tokens -> future tokens)

Input:
  - images: front_left (left), zed_left (mid), front_right (right)
  - text: prompt (Go straight / Turn left / Turn right / etc.)
  - vehicle state: speed, steer, yaw (best-effort from record)
  - past tokens: 6 tokens (past 3s, dt=0.5)

Output:
  - future tokens: 10 tokens (future 5s, dt=0.5)
  - tokens are <traj_XXXX> where XXXX in [0..2047]

JSONL example field:
  token_ctx_dt0p5_past3s_future5s: {
    tokens_past_3s: [..6 ints..],
    tokens_future_5s: [..10 ints..],
    ...
  }

Two LR groups:
  - base params: lr_base
  - traj params (input embeddings + lm_head): lr_traj

Modes:
  - full: train all parameters
  - traj_embed_head_only: freeze all, train only input_embeddings & lm_head
"""

import os
import io
import gc
import json
import math
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
# path utils
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

# -------------------------
# token/text helpers
# -------------------------
def tokens_to_traj_str(tokens: List[int], prefix: str = "traj") -> str:
    # join with space to ensure token boundaries
    return " ".join([f"<{prefix}_{t:04d}>" for t in tokens])

def find_subsequence(haystack: List[int], needle: List[int]) -> int:
    """return first index of needle in haystack, -1 if not found"""
    if len(needle) == 0:
        return 0
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1

# -------------------------
# state extract (best-effort)
# -------------------------
def get_speed_mps(rec: Dict[str, Any]) -> Optional[float]:
    # prefer _speed_mps
    if "_speed_mps" in rec:
        try:
            v = float(rec["_speed_mps"])
            if math.isfinite(v):
                return v
        except Exception:
            pass

    imu = rec.get("imu", {})
    if isinstance(imu, dict):
        nav807 = imu.get("807", {})
        if isinstance(nav807, dict) and "Vel" in nav807:
            try:
                v = float(nav807["Vel"])
                if math.isfinite(v):
                    return v
            except Exception:
                pass

    canfd = rec.get("canfd", {})
    if isinstance(canfd, dict) and "spd" in canfd:
        try:
            # NOTE: your canfd.spd might be km/h or wheel-based value; keep as raw
            v = float(canfd["spd"])
            if math.isfinite(v):
                return v
        except Exception:
            pass

    return None

def get_steer_deg(rec: Dict[str, Any]) -> Optional[float]:
    canfd = rec.get("canfd", {})
    if isinstance(canfd, dict) and "steer" in canfd:
        try:
            s = float(canfd["steer"])
            if math.isfinite(s):
                return s
        except Exception:
            pass
    return None

def get_yaw_deg(rec: Dict[str, Any]) -> Optional[float]:
    imu = rec.get("imu", {})
    if isinstance(imu, dict):
        nav810 = imu.get("810", {})
        if isinstance(nav810, dict) and "AngleHeading" in nav810:
            try:
                y = float(nav810["AngleHeading"])
                if math.isfinite(y):
                    return y
            except Exception:
                pass
    return None

# -------------------------
# Dataset: jsonl offsets + validation
# -------------------------
class JsonlOffsetDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        token_ctx_key: str,
        require_prompt: bool = False,
        past_len: int = 6,
        future_len: int = 10,
    ):
        self.jsonl_path = str(Path(jsonl_path).expanduser().resolve())
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(self.jsonl_path)

        self.token_ctx_key = token_ctx_key
        self.require_prompt = bool(require_prompt)
        self.past_len = int(past_len)
        self.future_len = int(future_len)

        self.offsets: List[int] = []
        self.bad_lines = 0

        # Scan once, keep only valid offsets (avoid collator crash).
        with open(self.jsonl_path, "rb") as f:
            off = 0
            for line in tqdm(f, desc="Scan jsonl"):
                raw = line.strip()
                if not raw:
                    off += len(line)
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8"))
                    if self.require_prompt:
                        p = obj.get("prompt", "")
                        if not isinstance(p, str) or len(p.strip()) == 0:
                            raise ValueError("empty_prompt")

                    ctx = obj.get(self.token_ctx_key, None)
                    if not isinstance(ctx, dict):
                        raise ValueError("missing_token_ctx")

                    tp = ctx.get("tokens_past_3s", None)
                    tf = ctx.get("tokens_future_5s", None)

                    if (not isinstance(tp, list)) or (not isinstance(tf, list)):
                        raise ValueError("bad_tokens_field_type")

                    if len(tp) != self.past_len or len(tf) != self.future_len:
                        raise ValueError("bad_tokens_length")

                    # all must be int in [0..2047] (loose check)
                    for t in tp + tf:
                        if not isinstance(t, int):
                            raise ValueError("token_not_int")
                        if t < 0:
                            raise ValueError("token_neg")

                    # paths check minimally
                    paths = obj.get("paths", {})
                    if not isinstance(paths, dict):
                        raise ValueError("missing_paths")

                    self.offsets.append(off)
                except Exception:
                    self.bad_lines += 1

                off += len(line)

        if len(self.offsets) == 0:
            raise RuntimeError(f"No valid samples found in: {self.jsonl_path}")

        print(f"[Dataset] valid={len(self.offsets)}, bad_lines_skipped={self.bad_lines}")

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
# Collator: 3 images + text prompt/state/past tokens -> labels on future tokens
# -------------------------
@dataclass
class DataCollatorQwenVLA3ImgTokens:
    processor: Any
    tokenizer: Any
    base_dir: str
    token_ctx_key: str
    prefix: str = "traj"
    max_length: int = 2048
    ignore_index: int = -100

    img_left_key: str = "front_left"
    img_mid_key: str = "zed_left"
    img_right_key: str = "front_right"

    # expected lengths
    past_len: int = 6
    future_len: int = 10

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images_batch: List[List[Image.Image]] = []
        full_texts: List[str] = []
        prompt_texts: List[str] = []

        for ex in batch:
            paths = ex.get("paths", {})
            if not isinstance(paths, dict):
                raise KeyError("paths must be dict")

            # image paths
            p_left = _join_base(self.base_dir, paths.get(self.img_left_key))
            p_mid = _join_base(self.base_dir, paths.get(self.img_mid_key))
            p_right = _join_base(self.base_dir, paths.get(self.img_right_key))

            for p in [p_left, p_mid, p_right]:
                if (p is None) or (not os.path.exists(p)):
                    raise FileNotFoundError(f"Image not found: {p}")

            img_left = Image.open(p_left).convert("RGB")
            img_mid = Image.open(p_mid).convert("RGB")
            img_right = Image.open(p_right).convert("RGB")
            images_batch.append([img_mid, img_left, img_right])  # order consistent with text description

            # prompt
            prompt = ex.get("prompt", "")
            if not isinstance(prompt, str):
                prompt = str(prompt)

            # vehicle state best-effort
            v = get_speed_mps(ex)
            steer = get_steer_deg(ex)
            yaw = get_yaw_deg(ex)

            state_parts = []
            if v is not None:
                state_parts.append(f"speed={v:.3f}")
            if steer is not None:
                state_parts.append(f"steer_deg={steer:.3f}")
            if yaw is not None:
                state_parts.append(f"heading_deg={yaw:.3f}")
            state_str = ", ".join(state_parts) if state_parts else "N/A"

            # tokens
            ctx = ex.get(self.token_ctx_key, {})
            tp = ctx.get("tokens_past_3s", [])
            tf = ctx.get("tokens_future_5s", [])

            if len(tp) != self.past_len or len(tf) != self.future_len:
                raise ValueError(f"Bad token length: past={len(tp)} future={len(tf)}")

            past_str = tokens_to_traj_str(tp, prefix=self.prefix)
            future_str = tokens_to_traj_str(tf, prefix=self.prefix)

            # user instruction
            user_text = (
                "You are an autonomous driving agent.\n"
                "You are given THREE camera images: middle, left, right.\n"
                f"Driving prompt (intent): {prompt}\n"
                f"Vehicle state: {state_str}\n"
                f"Past trajectory tokens (past 3.0s, dt=0.5s, 6 tokens): {past_str}\n\n"
                "Task: Predict the future trajectory tokens for the next 5.0s (dt=0.5s, 10 tokens).\n"
                "Output ONLY the 10 tokens, separated by spaces. Do not output any other words.\n"
            )

            # full messages (with assistant)
            mm_msgs_full = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_mid},   # middle
                        {"type": "image", "image": img_left},  # left
                        {"type": "image", "image": img_right}, # right
                        {"type": "text", "text": user_text},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": future_str}]},
            ]

            full_text = self.processor.apply_chat_template(
                mm_msgs_full, tokenize=False, add_generation_prompt=False
            )

            # prompt-only (NO assistant) -> used for masking labels
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

        # encode full with images
        model_inputs = self.processor(
            text=full_texts,
            images=images_batch,  # list[list[PIL]]
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # encode prompt text only (to locate boundary for masking)
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
                # fallback: mask by attention_mask length
                prompt_len = int(prompt_enc["attention_mask"][b].sum().item())
                labels[b, :prompt_len] = self.ignore_index
            else:
                labels[b, :start + len(prompt_ids)] = self.ignore_index

        model_inputs["labels"] = labels
        return model_inputs

# -------------------------
# Trainer with grouped LR + optional GC
# -------------------------
class GroupedLRTrainer(Trainer):
    def __init__(
        self,
        *args,
        lr_base: float,
        lr_traj: float,
        weight_decay: float,
        enable_gc: bool = False,
        gc_interval: int = 50,
        **kwargs
    ):
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

        in_emb = model.get_input_embeddings()
        out_emb = model.get_output_embeddings()

        traj_param_ptrs = set()
        if in_emb is not None and hasattr(in_emb, "weight"):
            traj_param_ptrs.add(in_emb.weight.data_ptr())
        if out_emb is not None and hasattr(out_emb, "weight"):
            traj_param_ptrs.add(out_emb.weight.data_ptr())

        base_decay, base_nodecay = [], []
        traj_decay, traj_nodecay = [], []

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue

            is_traj = hasattr(p, "data") and (p.data_ptr() in traj_param_ptrs)

            lname = n.lower()
            no_decay = ("bias" in lname) or ("layernorm" in lname) or ("layer_norm" in lname) or ("norm" in lname)

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

    # ✅ transformers 新版 training_step 多了 num_items_in_batch
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
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
    ap.add_argument("--processor_dir", default=None, help="If omitted, use model_dir.")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--prefix", type=str, default="traj")

    ap.add_argument("--token_ctx_key", type=str, default="token_ctx_dt0p5_past3s_future5s",
                    help="Key in each json line for token ctx dict.")
    ap.add_argument("--require_prompt", action="store_true")

    ap.add_argument("--img_left_key", type=str, default="front_left")
    ap.add_argument("--img_mid_key", type=str, default="zed_left")
    ap.add_argument("--img_right_key", type=str, default="front_right")

    ap.add_argument("--past_len", type=int, default=6)
    ap.add_argument("--future_len", type=int, default=10)

    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_base", type=float, default=1e-5)
    ap.add_argument("--lr_traj", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--train_mode", type=str, default="full",
                    choices=["full", "traj_embed_head_only"],
                    help="full: train all params; traj_embed_head_only: only train embeddings+lm_head.")
    ap.add_argument("--enable_gc", action="store_true")
    ap.add_argument("--gc_interval", type=int, default=50)

    ap.add_argument("--gradient_checkpointing", action="store_true")

    args = ap.parse_args()

    model_dir = str(Path(args.model_dir).expanduser().resolve())
    processor_dir = str(Path(args.processor_dir).expanduser().resolve()) if args.processor_dir else model_dir
    train_jsonl = str(Path(args.train_jsonl).expanduser().resolve())
    base_dir = str(Path(args.base_dir).expanduser().resolve())
    out_dir = str(Path(args.output_dir).expanduser().resolve())
    os.makedirs(out_dir, exist_ok=True)

    # explicit path checks (avoid HFValidationError when path is wrong)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"--model_dir not found: {model_dir}")
    if not os.path.isdir(processor_dir):
        raise FileNotFoundError(f"--processor_dir not found: {processor_dir}")
    if not os.path.isfile(train_jsonl):
        raise FileNotFoundError(f"--train_jsonl not found: {train_jsonl}")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"--base_dir not found: {base_dir}")

    print("[Paths]")
    print("  model_dir     :", model_dir)
    print("  processor_dir :", processor_dir)
    print("  train_jsonl   :", train_jsonl)
    print("  base_dir      :", base_dir)
    print("  output_dir    :", out_dir)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # processor/tokenizer (local only)
    processor = AutoProcessor.from_pretrained(
        processor_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = processor.tokenizer

    # model (local only)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto",
    )

    # resize embeddings if needed
    if getattr(model.config, "vocab_size", None) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # training mode
    if args.train_mode == "traj_embed_head_only":
        for p in model.parameters():
            p.requires_grad = False
        in_emb = model.get_input_embeddings()
        out_emb = model.get_output_embeddings()
        if in_emb is None or out_emb is None:
            raise RuntimeError("Cannot find input/output embeddings.")
        in_emb.weight.requires_grad = True
        out_emb.weight.requires_grad = True
        print("[Mode] traj_embed_head_only: only embeddings + lm_head trainable")
    else:
        for p in model.parameters():
            p.requires_grad = True
        print("[Mode] full: all parameters trainable")

    # dataset + collator
    ds = JsonlOffsetDataset(
        jsonl_path=train_jsonl,
        token_ctx_key=args.token_ctx_key,
        require_prompt=args.require_prompt,
        past_len=args.past_len,
        future_len=args.future_len,
    )

    collator = DataCollatorQwenVLA3ImgTokens(
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
        past_len=args.past_len,
        future_len=args.future_len,
    )

    bf16 = bool(torch.cuda.is_available())

    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr_base,  # unused for grouped optimizer (but keep)
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.log_every,
        save_steps=args.save_every,
        save_total_limit=2,
        bf16=bf16,
        fp16=False,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,   # MUST for multimodal
        report_to="none",
        optim="adamw_torch",           # will be overridden by create_optimizer()
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
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

    trainer.model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print("✅ saved to:", out_dir)

if __name__ == "__main__":
    main()
