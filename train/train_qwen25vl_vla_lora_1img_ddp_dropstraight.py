#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoRA-only training for Qwen2.5-VL VLA token prediction.
- Single main forward view (1 image) only.
- 4-GPU DDP via torchrun.
- Train ONLY LoRA adapters (base model weights not updated).
- Deterministically drop 2/3 of samples whose prompt == "Go straight" (DDP-safe).

Example:
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_qwen25vl_vla_lora_1img_ddp_dropstraight.py \
    --model_dir /home/lxh/lxh/12.26/qwen2_5vl_with_traj_tokens_2048 \
    --processor_dir /home/lxh/lxh/12.26/qwen2_5vl_with_traj_tokens_2048 \
    --train_jsonl /home/lxh/traj-vla/12.27-auto-vla/records_ctx_tokens_only_past3s_future5s_dt0p5.jsonl \
    --base_dir /home/lxh/lxh/vla_data \
    --output_dir /home/lxh/traj-vla/ckpt_lora_1img_ddp_dropstraight \
    --batch_size 1 --grad_accum 32 --epochs 1 \
    --lr 2e-4 --max_length 2048 \
    --img_main_key zed_left \
    --straight_prompt "Go straight" --straight_keep_prob 0.333333 --straight_hash_seed 123
"""

import os
import gc
import json
import math
import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from tqdm import tqdm
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

# PEFT (LoRA)
from peft import LoraConfig, get_peft_model, TaskType


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
    return " ".join([f"<{prefix}_{t:04d}>" for t in tokens])


def find_subsequence(haystack: List[int], needle: List[int]) -> int:
    if len(needle) == 0:
        return 0
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


# -------------------------
# state extract (best-effort)
# -------------------------
def get_speed_mps(rec: Dict[str, Any]) -> Optional[float]:
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
# Deterministic dropping helper (DDP-safe)
# -------------------------
def _stable_u01_from_key(key: str) -> float:
    # sha1 -> 160 bits, take first 8 bytes -> int -> [0,1)
    h = hashlib.sha1(key.encode("utf-8")).digest()
    x = int.from_bytes(h[:8], byteorder="big", signed=False)
    return (x % (10**12)) / float(10**12)


# -------------------------
# Dataset: jsonl offsets + validation + deterministic drop straight
# -------------------------
class JsonlOffsetDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        token_ctx_key: str,
        require_prompt: bool = False,
        past_len: int = 6,
        future_len: int = 10,
        # drop rule
        straight_prompt: str = "Go straight",
        straight_keep_prob: float = 1.0 / 3.0,   # keep 1/3 by default
        straight_hash_seed: int = 123,
    ):
        self.jsonl_path = str(Path(jsonl_path).expanduser().resolve())
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(self.jsonl_path)

        self.token_ctx_key = token_ctx_key
        self.require_prompt = bool(require_prompt)
        self.past_len = int(past_len)
        self.future_len = int(future_len)

        self.straight_prompt = str(straight_prompt)
        self.straight_keep_prob = float(straight_keep_prob)
        self.straight_hash_seed = int(straight_hash_seed)

        self.offsets: List[int] = []
        self.bad_lines = 0
        self.dropped_straight = 0

        with open(self.jsonl_path, "rb") as f:
            off = 0
            for line in tqdm(f, desc="Scan jsonl"):
                raw = line.strip()
                if not raw:
                    off += len(line)
                    continue

                try:
                    obj = json.loads(raw.decode("utf-8"))

                    # prompt check
                    p = obj.get("prompt", "")
                    if self.require_prompt:
                        if not isinstance(p, str) or len(p.strip()) == 0:
                            raise ValueError("empty_prompt")
                    if not isinstance(p, str):
                        p = str(p)

                    # deterministic drop for "Go straight"
                    if self.straight_keep_prob < 1.0 and p.strip() == self.straight_prompt:
                        # Use offset + seed + (optional id) to make decision stable
                        sid = obj.get("id", "")
                        key = f"{self.straight_hash_seed}|{off}|{sid}|{p}"
                        u = _stable_u01_from_key(key)
                        if u > self.straight_keep_prob:
                            self.dropped_straight += 1
                            off += len(line)
                            continue

                    ctx = obj.get(self.token_ctx_key, None)
                    if not isinstance(ctx, dict):
                        raise ValueError("missing_token_ctx")

                    tp = ctx.get("tokens_past_3s", None)
                    tf = ctx.get("tokens_future_5s", None)
                    if (not isinstance(tp, list)) or (not isinstance(tf, list)):
                        raise ValueError("bad_tokens_field_type")
                    if len(tp) != self.past_len or len(tf) != self.future_len:
                        raise ValueError("bad_tokens_length")

                    for t in tp + tf:
                        if not isinstance(t, int):
                            raise ValueError("token_not_int")
                        if t < 0:
                            raise ValueError("token_neg")

                    paths = obj.get("paths", {})
                    if not isinstance(paths, dict):
                        raise ValueError("missing_paths")

                    self.offsets.append(off)
                except Exception:
                    self.bad_lines += 1

                off += len(line)

        if len(self.offsets) == 0:
            raise RuntimeError(f"No valid samples found in: {self.jsonl_path}")

        print(
            f"[Dataset] valid={len(self.offsets)}, "
            f"bad_lines_skipped={self.bad_lines}, "
            f"dropped_straight={self.dropped_straight}, "
            f"straight_keep_prob={self.straight_keep_prob}"
        )

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
# Collator: 1 image + prompt/state/past tokens -> labels on future tokens
# -------------------------
@dataclass
class DataCollatorQwenVLA1ImgTokens:
    processor: Any
    tokenizer: Any
    base_dir: str
    token_ctx_key: str
    prefix: str = "traj"
    max_length: int = 2048
    ignore_index: int = -100

    img_main_key: str = "zed_left"   # 你要的“前向主视图”key（按你数据改）
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

            p_main = _join_base(self.base_dir, paths.get(self.img_main_key))
            if (p_main is None) or (not os.path.exists(p_main)):
                raise FileNotFoundError(f"Main image not found: {p_main} (key={self.img_main_key})")

            # 避免文件句柄泄露：用 with
            with Image.open(p_main) as im:
                img_main = im.convert("RGB")

            # processor 的多图接口需要 list[list[Image]]，这里单图也按这个结构
            images_batch.append([img_main])

            prompt = ex.get("prompt", "")
            if not isinstance(prompt, str):
                prompt = str(prompt)

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

            ctx = ex.get(self.token_ctx_key, {})
            tp = ctx.get("tokens_past_3s", [])
            tf = ctx.get("tokens_future_5s", [])

            if len(tp) != self.past_len or len(tf) != self.future_len:
                raise ValueError(f"Bad token length: past={len(tp)} future={len(tf)}")

            past_str = tokens_to_traj_str(tp, prefix=self.prefix)
            future_str = tokens_to_traj_str(tf, prefix=self.prefix)

            user_text = (
                "You are an autonomous driving agent.\n"
                "You are given ONE forward main-view camera image.\n"
                f"Driving prompt (intent): {prompt}\n"
                f"Vehicle state: {state_str}\n"
                f"Past trajectory tokens (past 3.0s, dt=0.5s, 6 tokens): {past_str}\n\n"
                "Task: Predict the future trajectory tokens for the next 5.0s (dt=0.5s, 10 tokens).\n"
                "Output ONLY the 10 tokens, separated by spaces. Do not output any other words.\n"
            )

            mm_msgs_full = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_main},
                        {"type": "text", "text": user_text},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": future_str}]},
            ]
            full_text = self.processor.apply_chat_template(
                mm_msgs_full, tokenize=False, add_generation_prompt=False
            )

            mm_msgs_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_main},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
            prompt_text = self.processor.apply_chat_template(
                mm_msgs_prompt, tokenize=False, add_generation_prompt=True
            )

            full_texts.append(full_text)
            prompt_texts.append(prompt_text)

        model_inputs = self.processor(
            text=full_texts,
            images=images_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

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
# Trainer: add optional GC
# -------------------------
class SimpleTrainer(Trainer):
    def __init__(self, *args, enable_gc: bool = False, gc_interval: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self._enable_gc = bool(enable_gc)
        self._gc_interval = int(gc_interval)

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        if self._enable_gc and (self.state.global_step % self._gc_interval == 0):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return loss


def _print_trainable_params(model: torch.nn.Module):
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100.0 * trainable / max(1, total)
    print(f"[Params] trainable={trainable:,} / total={total:,} ({pct:.4f}%)")


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

    ap.add_argument("--prefix", type=str, default="traj")
    ap.add_argument("--token_ctx_key", type=str, default="token_ctx_dt0p5_past3s_future5s")
    ap.add_argument("--require_prompt", action="store_true")

    # single main view key
    ap.add_argument("--img_main_key", type=str, default="zed_left")

    ap.add_argument("--past_len", type=int, default=6)
    ap.add_argument("--future_len", type=int, default=10)

    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--enable_gc", action="store_true")
    ap.add_argument("--gc_interval", type=int, default=50)
    ap.add_argument("--gradient_checkpointing", action="store_true")

    # LoRA config
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names."
    )

    # drop straight
    ap.add_argument("--straight_prompt", type=str, default="Go straight")
    ap.add_argument("--straight_keep_prob", type=float, default=1.0/3.0, help="Keep prob for straight samples.")
    ap.add_argument("--straight_hash_seed", type=int, default=123)

    # DDP niceties
    ap.add_argument("--ddp_find_unused_parameters", action="store_true",
                    help="If training hangs, try enabling this. Default False is usually faster/safer for LoRA.")

    args = ap.parse_args()

    # Paths
    model_dir = str(Path(args.model_dir).expanduser().resolve())
    processor_dir = str(Path(args.processor_dir).expanduser().resolve()) if args.processor_dir else model_dir
    train_jsonl = str(Path(args.train_jsonl).expanduser().resolve())
    base_dir = str(Path(args.base_dir).expanduser().resolve())
    out_dir = str(Path(args.output_dir).expanduser().resolve())
    os.makedirs(out_dir, exist_ok=True)

    print("[Paths]")
    print("  model_dir     :", model_dir)
    print("  processor_dir :", processor_dir)
    print("  train_jsonl   :", train_jsonl)
    print("  base_dir      :", base_dir)
    print("  output_dir    :", out_dir)

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    processor = AutoProcessor.from_pretrained(
        processor_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = processor.tokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # IMPORTANT for DDP:
    # - do NOT use device_map="auto" here (it may shard / confuse DDP)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        device_map=None,
    )

    # resize vocab if needed (your traj tokens)
    if getattr(model.config, "vocab_size", None) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # --- Apply LoRA (LoRA-only training) ---
    target_modules = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    print("[LoRA] target_modules =", target_modules)
    _print_trainable_params(model)

    # Dataset with deterministic drop
    ds = JsonlOffsetDataset(
        jsonl_path=train_jsonl,
        token_ctx_key=args.token_ctx_key,
        require_prompt=args.require_prompt,
        past_len=args.past_len,
        future_len=args.future_len,
        straight_prompt=args.straight_prompt,
        straight_keep_prob=args.straight_keep_prob,
        straight_hash_seed=args.straight_hash_seed,
    )

    collator = DataCollatorQwenVLA1ImgTokens(
        processor=processor,
        tokenizer=tokenizer,
        base_dir=base_dir,
        token_ctx_key=args.token_ctx_key,
        prefix=args.prefix,
        max_length=args.max_length,
        ignore_index=-100,
        img_main_key=args.img_main_key,
        past_len=args.past_len,
        future_len=args.future_len,
    )

    bf16 = bool(torch.cuda.is_available())

    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.log_every,
        save_steps=args.save_every,
        save_total_limit=2,
        bf16=bf16,
        fp16=False,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        ddp_find_unused_parameters=bool(args.ddp_find_unused_parameters),
    )

    trainer = SimpleTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        enable_gc=args.enable_gc,
        gc_interval=args.gc_interval,
    )

    trainer.train()

    # Save LoRA adapter
    trainer.model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print("✅ LoRA adapter + processor saved to:", out_dir)
    print("Tip: You can later merge LoRA into base model if needed.")


if __name__ == "__main__":
    main()
