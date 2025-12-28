#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grouped-LR full finetune for Qwen2.5-VL (3 images: left/mid/right) to predict future traj tokens.

Data JSONL example (per line):
{
  "paths": {"front_left": "...jpg", "zed_left": "...jpg", "front_right": "...jpg", ...},
  "prompt": "Go straight",   # MUST
  "_speed_mps": 4.37, ...
  "token_ctx_dt0p5_past3s_future5s": {
      "tokens_past_3s": [int]*6,
      "tokens_future_5s":[int]*10
  }
}

Training:
- param group A: embeddings/lm_head weights -> lr_traj, but grad masked to ONLY <traj_0000..2047> rows
- param group B: all other parameters -> lr_base (full finetune)

Why:
- prevents base model from being destroyed by large LR needed for newly-added traj tokens
- still allows end-to-end learning

Usage example:
python train_qwen25vl_vla_grouped_lr_3img.py \
  --model_dir "/home/lxh/vla/12.27-auto-vla/qwen2_5vl3b_with_traj2048" \
  --train_jsonl "/home/lxh/vla/12.27-auto-vla/records_ctx_tokens_only_past3s_future5s_dt0p5.jsonl" \
  --base_dir "/media/lxh/My Passport/vla_data" \
  --output_dir "/home/lxh/vla/12.27-auto-vla/ckpt_grouped_full" \
  --lr_base 1e-5 --lr_traj 1e-4 --batch_size 1 --grad_accum 32 --epochs 1
"""

import os
import json
import math
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# -------------------------
# path utils
# -------------------------
def is_abs(p: str) -> bool:
    try:
        return os.path.isabs(p)
    except Exception:
        return False

def join_base(base_dir: str, p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    p = str(p)
    return p if is_abs(p) else os.path.join(base_dir, p)

def find_subsequence(haystack: List[int], needle: List[int]) -> int:
    """return first index of needle in haystack, -1 if not found"""
    if len(needle) == 0:
        return 0
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


# -------------------------
# traj token helpers
# -------------------------
def traj_token_str(tid: int, prefix: str = "traj") -> str:
    return f"<{prefix}_{int(tid):04d}>"

def build_traj_token_ids(tokenizer, K: int, prefix: str) -> List[int]:
    toks = [traj_token_str(i, prefix=prefix) for i in range(K)]
    ids = tokenizer.convert_tokens_to_ids(toks)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    bad = []
    for i, tid in enumerate(ids):
        if tid is None or (unk_id is not None and tid == unk_id):
            bad.append((i, toks[i], tid))
    if bad:
        raise ValueError(f"Missing traj tokens in tokenizer vocab. examples={bad[:10]}")
    return ids

def apply_row_grad_mask(weight: torch.Tensor, row_ids: torch.Tensor):
    """
    Only allow gradient updates on specified rows. Others zeroed.
    weight: [V, D] 2D
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight, got {tuple(weight.shape)}")

    V = weight.shape[0]
    device = weight.device
    mask = torch.zeros((V, 1), device=device, dtype=weight.dtype)
    mask[row_ids] = 1.0

    def hook(grad):
        return grad * mask

    weight.register_hook(hook)


# -------------------------
# safe getter + vehicle state text
# -------------------------
def safe_get(d: Any, keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def build_vehicle_state_text(ex: Dict[str, Any]) -> str:
    spd_mps = ex.get("_speed_mps", None)
    steer_deg = safe_get(ex, ["canfd", "steer"], None)
    wheel_spd = safe_get(ex, ["canfd", "spd"], None)
    tq = safe_get(ex, ["canfd", "tq"], None)

    yaw_rate = safe_get(ex, ["imu", "801", "AngRateRawZ"], None)
    accel_x = safe_get(ex, ["imu", "802", "AccelRawX"], None)
    accel_y = safe_get(ex, ["imu", "802", "AccelRawY"], None)

    parts = []
    if spd_mps is not None:
        parts.append(f"speed_mps={float(spd_mps):.2f}")
    if wheel_spd is not None:
        parts.append(f"wheel_spd={float(wheel_spd):.2f}")
    if steer_deg is not None:
        parts.append(f"steer_deg={float(steer_deg):.2f}")
    if tq is not None:
        parts.append(f"drive_tq={float(tq):.1f}")
    if yaw_rate is not None:
        parts.append(f"yaw_rate={float(yaw_rate):.3f}")
    if accel_x is not None and accel_y is not None:
        parts.append(f"accel_xy=({float(accel_x):.4f},{float(accel_y):.4f})")

    return "vehicle_state: " + (", ".join(parts) if parts else "(missing)")


# -------------------------
# dataset: prefilter lines requiring prompt+tokens+paths
# -------------------------
class JsonlOffsetDatasetFiltered(Dataset):
    """
    Build offsets once; keep only lines that satisfy required fields
    (so training won't crash mid-epoch due to bad records).
    """
    def __init__(
        self,
        jsonl_path: str,
        require_prompt: bool = True,
        token_ctx_key: str = "token_ctx_dt0p5_past3s_future5s",
        img_keys: Tuple[str, str, str] = ("front_left", "zed_left", "front_right"),
    ):
        self.jsonl_path = str(Path(jsonl_path).expanduser().resolve())
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(self.jsonl_path)

        self.require_prompt = require_prompt
        self.token_ctx_key = token_ctx_key
        self.img_keys = img_keys

        self.offsets: List[int] = []
        self.bad_stats = {
            "empty_line": 0,
            "json_error": 0,
            "missing_prompt": 0,
            "missing_paths": 0,
            "missing_images": 0,
            "missing_tokens": 0,
            "bad_token_len": 0,
        }

        with open(self.jsonl_path, "rb") as f:
            off = 0
            for line in tqdm(f, desc="Index jsonl"):
                raw = line.strip()
                if not raw:
                    self.bad_stats["empty_line"] += 1
                    off += len(line)
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:
                    self.bad_stats["json_error"] += 1
                    off += len(line)
                    continue

                if self.require_prompt:
                    p = obj.get("prompt", None)
                    if p is None or (isinstance(p, str) and p.strip() == ""):
                        self.bad_stats["missing_prompt"] += 1
                        off += len(line)
                        continue

                paths = obj.get("paths", None)
                if not isinstance(paths, dict):
                    self.bad_stats["missing_paths"] += 1
                    off += len(line)
                    continue

                ok_imgs = True
                for k in self.img_keys:
                    if k not in paths:
                        ok_imgs = False
                        break
                if not ok_imgs:
                    self.bad_stats["missing_images"] += 1
                    off += len(line)
                    continue

                ctx = obj.get(self.token_ctx_key, None)
                if not isinstance(ctx, dict):
                    self.bad_stats["missing_tokens"] += 1
                    off += len(line)
                    continue

                past = ctx.get("tokens_past_3s", None)
                fut = ctx.get("tokens_future_5s", None)
                if not isinstance(past, list) or not isinstance(fut, list):
                    self.bad_stats["missing_tokens"] += 1
                    off += len(line)
                    continue
                if len(past) != 6 or len(fut) != 10:
                    self.bad_stats["bad_token_len"] += 1
                    off += len(line)
                    continue

                self.offsets.append(off)
                off += len(line)

        if not self.offsets:
            raise RuntimeError(f"No valid samples found in {self.jsonl_path}. stats={self.bad_stats}")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        off = self.offsets[idx]
        with open(self.jsonl_path, "rb") as f:
            f.seek(off)
            line = f.readline()
        return json.loads(line.decode("utf-8"))


# -------------------------
# collator: 3 images + (prompt + state + past tokens) -> future tokens
# -------------------------
@dataclass
class DataCollatorVLA3Img:
    processor: Any
    tokenizer: Any
    base_dir: str
    max_length: int = 2048
    ignore_index: int = -100

    img_left_key: str = "front_left"
    img_mid_key: str = "zed_left"
    img_right_key: str = "front_right"

    token_ctx_key: str = "token_ctx_dt0p5_past3s_future5s"
    prefix: str = "traj"

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images_batch: List[List[Image.Image]] = []
        full_texts: List[str] = []
        prompt_texts: List[str] = []

        for ex in batch:
            # ---- MUST HAVE PROMPT ----
            prompt = ex.get("prompt", None)
            if prompt is None or (isinstance(prompt, str) and prompt.strip() == ""):
                raise ValueError("This training requires ex['prompt'] (e.g., 'Go straight' / 'Turn left').")

            paths = ex.get("paths", {})
            pL = join_base(self.base_dir, paths.get(self.img_left_key))
            pM = join_base(self.base_dir, paths.get(self.img_mid_key))
            pR = join_base(self.base_dir, paths.get(self.img_right_key))

            for name, p in [("left", pL), ("mid", pM), ("right", pR)]:
                if (p is None) or (not os.path.exists(p)):
                    raise FileNotFoundError(f"{name} image not found: {p}")

            imgL = Image.open(pL).convert("RGB")
            imgM = Image.open(pM).convert("RGB")
            imgR = Image.open(pR).convert("RGB")

            # keep consistent ordering between chat template and images_batch
            # Here: middle, left, right
            images_batch.append([imgM, imgL, imgR])

            ctx = ex.get(self.token_ctx_key, {})
            past_ids = ctx.get("tokens_past_3s")
            fut_ids = ctx.get("tokens_future_5s")
            if len(past_ids) != 6 or len(fut_ids) != 10:
                raise ValueError(f"Bad token lengths: past={len(past_ids)} future={len(fut_ids)}")

            past_tokens = " ".join(traj_token_str(t, prefix=self.prefix) for t in past_ids)
            future_tokens = " ".join(traj_token_str(t, prefix=self.prefix) for t in fut_ids)

            state_text = build_vehicle_state_text(ex)

            # ---- IMPORTANT: prompt (straight/turn) is explicitly in input ----
            user_text = (
                "You are an autonomous driving agent.\n"
                "You are given three camera images (middle/left/right), the vehicle state, "
                "and past trajectory tokens.\n"
                "Task: Predict the future trajectory tokens for the next 5.0 seconds.\n\n"
                f"Driving prompt (MUST): {prompt}\n"
                f"{state_text}\n"
                f"Past 3.0s trajectory tokens (6): {past_tokens}\n\n"
                "Output ONLY the future trajectory tokens (10 tokens), separated by spaces.\n"
                "Format: <traj_XXXX> <traj_XXXX> ... (10 tokens)\n"
            )

            mm_msgs_full = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": imgM},  # mid
                        {"type": "image", "image": imgL},  # left
                        {"type": "image", "image": imgR},  # right
                        {"type": "text", "text": user_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": future_tokens}],
                },
            ]
            full_text = self.processor.apply_chat_template(
                mm_msgs_full, tokenize=False, add_generation_prompt=False
            )

            mm_msgs_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": imgM},
                        {"type": "image", "image": imgL},
                        {"type": "image", "image": imgR},
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

        # tokenize prompt-only (text) to locate/mask prefix tokens
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
            p_ids = prompt_enc["input_ids"][b].tolist()

            pad_id = self.tokenizer.pad_token_id
            if pad_id is not None:
                while len(p_ids) > 0 and p_ids[-1] == pad_id:
                    p_ids.pop()

            start = find_subsequence(full_ids, p_ids)
            if start == -1:
                prompt_len = int(prompt_enc["attention_mask"][b].sum().item())
                labels[b, :prompt_len] = self.ignore_index
            else:
                labels[b, :start + len(p_ids)] = self.ignore_index

        model_inputs["labels"] = labels
        return model_inputs


# -------------------------
# metrics (optional)
# -------------------------
@torch.no_grad()
def token_acc_over_traj_positions(logits: torch.Tensor, labels: torch.Tensor, traj_token_set: set) -> float:
    pred = torch.argmax(logits, dim=-1)  # [B,S]
    mask = labels != -100
    if mask.sum().item() == 0:
        return float("nan")

    traj_mask = torch.zeros_like(mask)
    # Only count positions where GT is traj token
    for tid in traj_token_set:
        traj_mask |= (labels == tid)
    mask = mask & traj_mask

    denom = mask.sum().item()
    if denom == 0:
        return float("nan")
    correct = ((pred == labels) & mask).sum().item()
    return float(correct) / float(denom)


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--prefix", type=str, default="traj")

    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)

    # ---- grouped LR ----
    ap.add_argument("--lr_base", type=float, default=1e-5, help="LR for all base params (full finetune).")
    ap.add_argument("--lr_traj", type=float, default=1e-4, help="LR for embeddings/lm_head (traj rows masked).")

    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=1000)

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--require_prompt", action="store_true", help="Require prompt field in each sample (recommended).")
    ap.add_argument("--enable_gc", action="store_true", help="Enable gradient checkpointing to save VRAM.")

    # image keys (your exact requirement)
    ap.add_argument("--img_left_key", type=str, default="front_left")
    ap.add_argument("--img_mid_key", type=str, default="zed_left")
    ap.add_argument("--img_right_key", type=str, default="front_right")

    ap.add_argument("--token_ctx_key", type=str, default="token_ctx_dt0p5_past3s_future5s")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_dir = str(Path(args.model_dir).expanduser().resolve())
    train_jsonl = str(Path(args.train_jsonl).expanduser().resolve())
    base_dir = str(Path(args.base_dir).expanduser().resolve())
    out_dir = str(Path(args.output_dir).expanduser().resolve())
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and (torch.cuda.get_device_capability(0)[0] >= 8)

    processor = AutoProcessor.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32),
    ).to(device)

    # vocab align
    if getattr(model.config, "vocab_size", None) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    model.config.use_cache = False
    if args.enable_gc:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # build traj ids & set
    traj_token_ids = build_traj_token_ids(tokenizer, K=args.K, prefix=args.prefix)
    traj_token_set = set(traj_token_ids)

    # ----- define traj params (embeddings/lm_head) + grad row mask -----
    in_emb = model.get_input_embeddings()
    out_emb = model.get_output_embeddings()
    if in_emb is None or out_emb is None:
        raise RuntimeError("Model embeddings not found.")

    # Ensure all params are trainable (full finetune)
    for p in model.parameters():
        p.requires_grad = True

    # Apply row mask on embeddings & lm_head, so only <traj_*> rows change
    row_ids_in = torch.tensor(sorted(traj_token_ids), dtype=torch.long, device=in_emb.weight.device)
    apply_row_grad_mask(in_emb.weight, row_ids_in)

    # handle tied weights
    tied = (out_emb.weight is in_emb.weight)
    if not tied:
        row_ids_out = torch.tensor(sorted(traj_token_ids), dtype=torch.long, device=out_emb.weight.device)
        apply_row_grad_mask(out_emb.weight, row_ids_out)

    # build param groups safely (avoid duplicates)
    traj_params = []
    traj_param_ids = set()

    def add_param_once(p):
        pid = id(p)
        if pid not in traj_param_ids:
            traj_params.append(p)
            traj_param_ids.add(pid)

    add_param_once(in_emb.weight)
    if not tied:
        add_param_once(out_emb.weight)

    base_params = []
    base_param_ids = set()
    traj_ids_set = set(id(p) for p in traj_params)

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in traj_ids_set:
            continue
        pid = id(p)
        if pid not in base_param_ids:
            base_params.append(p)
            base_param_ids.add(pid)

    if len(base_params) == 0:
        raise RuntimeError("No base params collected for full finetune (unexpected).")

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": args.lr_base, "weight_decay": args.weight_decay},
            {"params": traj_params, "lr": args.lr_traj, "weight_decay": 0.0},
        ]
    )

    # dataset + dataloader
    ds = JsonlOffsetDatasetFiltered(
        train_jsonl,
        require_prompt=args.require_prompt or True,  # strongly recommended
        token_ctx_key=args.token_ctx_key,
        img_keys=(args.img_left_key, args.img_mid_key, args.img_right_key),
    )

    collator = DataCollatorVLA3Img(
        processor=processor,
        tokenizer=tokenizer,
        base_dir=base_dir,
        max_length=args.max_length,
        ignore_index=-100,
        img_left_key=args.img_left_key,
        img_mid_key=args.img_mid_key,
        img_right_key=args.img_right_key,
        token_ctx_key=args.token_ctx_key,
        prefix=args.prefix,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
        drop_last=False,
    )

    # schedule: cosine with warmup on optimizer steps
    steps_per_epoch = math.ceil(len(dl) / max(1, args.grad_accum))
    total_opt_steps = int(math.ceil(steps_per_epoch * args.epochs))
    warmup_steps = int(args.warmup_ratio * total_opt_steps)

    def lr_mult(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        prog = float(step - warmup_steps) / float(max(1, total_opt_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scaler = None
    if torch.cuda.is_available() and (not use_bf16):
        scaler = torch.cuda.amp.GradScaler()

    # logging
    run_cfg = {
        "model_dir": model_dir,
        "train_jsonl": train_jsonl,
        "base_dir": base_dir,
        "output_dir": out_dir,
        "lr_base": args.lr_base,
        "lr_traj": args.lr_traj,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "warmup_ratio": args.warmup_ratio,
        "total_opt_steps": total_opt_steps,
        "warmup_steps": warmup_steps,
        "use_bf16": bool(use_bf16),
        "enable_gc": bool(args.enable_gc),
        "dataset_bad_stats": ds.bad_stats,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)

    model.train()
    global_iter = 0
    opt_step = 0
    last_save_t = time.time()

    pbar = tqdm(total=int(len(dl) * args.epochs), desc="train(iters)")
    epoch_float = 0.0

    for ep in range(int(math.ceil(args.epochs))):
        for batch in dl:
            global_iter += 1
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=autocast_dtype):
                out = model(**batch, return_dict=True)
                loss = out.loss / float(args.grad_accum)

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            pbar.update(1)

            if global_iter % args.grad_accum == 0:
                # update LR (both groups) with same multiplier
                m = lr_mult(opt_step)
                optimizer.param_groups[0]["lr"] = args.lr_base * m
                optimizer.param_groups[1]["lr"] = args.lr_traj * m

                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % args.log_every == 0:
                    with torch.no_grad():
                        acc = token_acc_over_traj_positions(out.logits.detach(), batch["labels"], traj_token_set)
                    tqdm.write(
                        f"[opt_step {opt_step}/{total_opt_steps}] "
                        f"loss={float(loss.item()*args.grad_accum):.4f} "
                        f"lr_base={optimizer.param_groups[0]['lr']:.2e} "
                        f"lr_traj={optimizer.param_groups[1]['lr']:.2e} "
                        f"traj_acc={acc}"
                    )

                if opt_step % args.save_every == 0:
                    ckpt = os.path.join(out_dir, f"ckpt_step{opt_step}")
                    os.makedirs(ckpt, exist_ok=True)
                    model.save_pretrained(ckpt)
                    processor.save_pretrained(ckpt)
                    tqdm.write(f"✅ saved: {ckpt}")

            # stop when reaching fractional epochs
            epoch_float = (ep + global_iter / max(1, len(dl)))
            if epoch_float >= args.epochs:
                break

        if epoch_float >= args.epochs:
            break

    pbar.close()

    # final save
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print("✅ Final saved to:", out_dir)
    print("Dataset bad stats:", ds.bad_stats)
    print("NOTE: input prompt is REQUIRED (straight/turn). It is included as 'Driving prompt (MUST): ...' in the instruction.")


if __name__ == "__main__":
    main()
