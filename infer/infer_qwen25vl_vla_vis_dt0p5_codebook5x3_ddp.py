#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_qwen25vl_vla_vis_dt0p5_codebook5x3_ddp.py

4-card inference (data-parallel via torchrun): each rank handles a shard of samples,
runs Qwen2.5-VL VLA generation, decodes 10 traj tokens, reconstructs XY via codebook
(5x3 per token, 10 tokens => 50 steps), and visualizes:
- history trajectory (past 3s = 6 tokens = 30 steps)
- GT future trajectory (future 5s = 10 tokens = 50 steps)
- Pred future trajectory (10 tokens)

Launch (4 GPUs):
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --nproc_per_node=4 infer_qwen25vl_vla_vis_dt0p5_codebook5x3_ddp.py \
    --model_dir "/home/lxh/traj-vla/12.27-auto-vla/ckpt_grouped_full" \
    --processor_dir "/home/lxh/traj-vla/12.27-auto-vla/ckpt_grouped_full" \
    --jsonl "/home/lxh/traj-vla/12.27-auto-vla/records_ctx_tokens_only_past3s_future5s_dt0p5.jsonl" \
    --base_dir "/home/lxh/lxh/vla_data" \
    --codebook_json "/home/lxh/traj-vla/12.27-auto-vla/action_codebook_future5frames_kmeans2048_dt0p5.json" \
    --out_dir "/home/lxh/traj-vla/12.27-auto-vla/infer_vis_ddp" \
    --num_samples 2000 --start_idx 0 --require_prompt

Notes:
- This script uses "single-GPU per rank" model placement (no device_map="auto"),
  which is the correct approach for torchrun data-parallel inference.
- If you want "single sample model-parallel across GPUs", do NOT use torchrun; use device_map="auto" instead.
"""

import os
import re
import gc
import json
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps

import torch
import torch.distributed as dist

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# -------------------------
# utils: early parse cuda_visible_devices (optional)
# -------------------------
def _early_parse_cuda_visible_devices() -> Optional[str]:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--cuda_visible_devices", type=str, default=None)
    args, _ = ap.parse_known_args()
    return args.cuda_visible_devices

_cuda_vis = _early_parse_cuda_visible_devices()
if _cuda_vis is not None:
    # Only set if user explicitly provides; safe for torchrun when you pass same arg to all ranks
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_vis


# -------------------------
# distributed init
# -------------------------
def ddp_init() -> Tuple[bool, int, int, int]:
    """
    Returns: (is_ddp, rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return True, rank, world, local_rank
    return False, 0, 1, 0


def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def ddp_destroy():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


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
# image utils
# -------------------------
def open_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


# -------------------------
# token/text helpers
# -------------------------
def tokens_to_traj_str(tokens: List[int], prefix: str = "traj") -> str:
    return " ".join([f"<{prefix}_{t:04d}>" for t in tokens])

_TRAJ_RE_CACHE: Dict[str, re.Pattern] = {}

def parse_traj_tokens_from_text(text: str, prefix: str = "traj") -> List[int]:
    """
    Extract ids from occurrences like <traj_0123>.
    """
    if prefix not in _TRAJ_RE_CACHE:
        _TRAJ_RE_CACHE[prefix] = re.compile(rf"<{re.escape(prefix)}_(\d{{4}})>")
    pat = _TRAJ_RE_CACHE[prefix]
    out = []
    for m in pat.finditer(text):
        try:
            out.append(int(m.group(1)))
        except Exception:
            pass
    return out


# -------------------------
# state extract (same as your trainer)
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
# Dataset: jsonl offsets + validation (reuse your logic)
# -------------------------
class JsonlOffsetDataset:
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
# Codebook loader (JSON centers): token -> 15D -> 5x3
# -------------------------
class Codebook5x3:
    def __init__(self, codebook_json_path: str):
        p = Path(codebook_json_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(str(p))
        with p.open("r", encoding="utf-8") as f:
            cb = json.load(f)

        if "centers_unscaled_for_reconstruction" not in cb:
            raise KeyError("codebook json missing centers_unscaled_for_reconstruction")

        centers = np.asarray(cb["centers_unscaled_for_reconstruction"], dtype=np.float32)  # [K,15]
        if centers.ndim != 2 or centers.shape[1] != 15:
            raise ValueError(f"Bad centers shape: {centers.shape}, expected [K,15]")

        self.name = cb.get("name", p.name)
        self.K = int(cb.get("k", centers.shape[0]))
        self.chunk_steps = int(cb.get("chunk_steps", 5))
        self.dt_token = float(cb.get("dt_token", cb.get("dt_token", cb.get("dt_token", 0.5))))
        self.hz_src = float(cb.get("hz_src", 10.0))

        if self.chunk_steps != 5:
            raise ValueError(f"Expected chunk_steps=5, got {self.chunk_steps}")
        if self.K != centers.shape[0]:
            # tolerate mismatch but warn
            print(f"[WARN] codebook k={self.K} but centers.shape[0]={centers.shape[0]}, using centers.shape[0].")
            self.K = centers.shape[0]

        self.centers = centers

    def token_to_5x3(self, tid: int) -> np.ndarray:
        if tid < 0 or tid >= self.K:
            # unknown token -> zeros
            return np.zeros((5, 3), dtype=np.float32)
        v15 = self.centers[tid]  # [15]
        return v15.reshape(5, 3)

    def tokens_to_steps_xy(self, tids: List[int]) -> np.ndarray:
        """
        Convert token sequence -> per-step dx,dy array of shape [len(tids)*5, 2].
        All dx,dy are already in ego@current frame (per your codebook construction).
        """
        steps = []
        for tid in tids:
            v = self.token_to_5x3(int(tid))  # [5,3]
            steps.append(v[:, 0:2])          # [5,2]
        if len(steps) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.concatenate(steps, axis=0).astype(np.float32)


def steps_to_xy(steps_xy: np.ndarray, start_at_origin: bool = True, end_at_origin: bool = False) -> np.ndarray:
    """
    steps_xy: [T,2] is per-step delta x,y.
    Returns points xy: [T+1,2] cumulative positions.

    - start_at_origin=True: xy[0]=(0,0)
    - end_at_origin=True: shift so last point becomes (0,0) (useful for past trajectory ending at current time)
    """
    if steps_xy.size == 0:
        return np.zeros((1, 2), dtype=np.float32)

    xy = np.zeros((steps_xy.shape[0] + 1, 2), dtype=np.float32)
    if not start_at_origin:
        xy[0] = 0.0
    xy[1:] = np.cumsum(steps_xy, axis=0)

    if end_at_origin:
        xy = xy - xy[-1][None, :]
    return xy


# -------------------------
# Build inference prompt (match training)
# -------------------------
def build_user_text(
    ex: Dict[str, Any],
    token_ctx_key: str,
    prefix: str,
    past_len: int,
) -> Tuple[str, List[int]]:
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

    ctx = ex.get(token_ctx_key, {})
    tp = ctx.get("tokens_past_3s", [])
    if (not isinstance(tp, list)) or len(tp) != past_len:
        raise ValueError(f"Bad past tokens in sample: got {type(tp)} len={len(tp) if isinstance(tp,list) else 'NA'}")

    past_str = tokens_to_traj_str([int(x) for x in tp], prefix=prefix)

    user_text = (
        "You are an autonomous driving agent.\n"
        "You are given THREE camera images: middle, left, right.\n"
        f"Driving prompt (intent): {prompt}\n"
        f"Vehicle state: {state_str}\n"
        f"Past trajectory tokens (past 3.0s, dt=0.5s, 6 tokens): {past_str}\n\n"
        "Task: Predict the future trajectory tokens for the next 5.0s (dt=0.5s, 10 tokens).\n"
        "Output ONLY the 10 tokens, separated by spaces. Do not output any other words.\n"
    )
    return user_text, [int(x) for x in tp]


# -------------------------
# Visualization
# -------------------------
def plot_xy_compare(
    out_png: str,
    past_xy: np.ndarray,         # [31,2] ends at (0,0)
    fut_gt_xy: np.ndarray,       # [51,2] starts at (0,0)
    fut_pred_xy: np.ndarray,     # [51,2] starts at (0,0)
    title: str,
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 7))
    plt.plot(past_xy[:, 0], past_xy[:, 1], marker=".", linewidth=1.5, label="past (history)")
    plt.plot(fut_gt_xy[:, 0], fut_gt_xy[:, 1], marker=".", linewidth=1.5, label="future GT")
    plt.plot(fut_pred_xy[:, 0], fut_pred_xy[:, 1], marker=".", linewidth=1.5, label="future Pred")

    plt.scatter([0.0], [0.0], s=60, marker="x", label="t=0 (ego)")

    plt.axis("equal")
    plt.grid(True, linewidth=0.5)
    plt.xlabel("x (forward, m)")
    plt.ylabel("y (left, m)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda_visible_devices", type=str, default=None,
                    help='Optional: set CUDA_VISIBLE_DEVICES inside script (e.g. "0,1,2,3"). Prefer setting env outside.')
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--processor_dir", default=None)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--codebook_json", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--token_ctx_key", type=str, default="token_ctx_dt0p5_past3s_future5s")
    ap.add_argument("--prefix", type=str, default="traj")

    ap.add_argument("--img_left_key", type=str, default="front_left")
    ap.add_argument("--img_mid_key", type=str, default="zed_left")
    ap.add_argument("--img_right_key", type=str, default="front_right")

    ap.add_argument("--past_len", type=int, default=6)
    ap.add_argument("--future_len", type=int, default=10)

    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--num_samples", type=int, default=50)

    ap.add_argument("--require_prompt", action="store_true")

    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force_fp16", action="store_true", help="Use fp16 instead of bf16 if your GPU/stack prefers.")
    ap.add_argument("--empty_cache_every", type=int, default=20)

    args = ap.parse_args()

    # DDP init (torchrun)
    is_ddp, rank, world, local_rank = ddp_init()

    # Print device info only on rank0
    if rank == 0:
        print(f"[DDP] is_ddp={is_ddp} world={world}")
        print(f"[CUDA_VISIBLE_DEVICES] {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        if torch.cuda.is_available():
            print(f"[CUDA] visible device count = {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - cuda:{i} = {torch.cuda.get_device_name(i)}")

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    model_dir = str(Path(args.model_dir).expanduser().resolve())
    processor_dir = str(Path(args.processor_dir).expanduser().resolve()) if args.processor_dir else model_dir
    jsonl_path = str(Path(args.jsonl).expanduser().resolve())
    base_dir = str(Path(args.base_dir).expanduser().resolve())
    codebook_json = str(Path(args.codebook_json).expanduser().resolve())
    out_dir = str(Path(args.out_dir).expanduser().resolve())

    # per-rank output
    out_rank_dir = os.path.join(out_dir, f"rank{rank}")
    os.makedirs(out_rank_dir, exist_ok=True)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"--model_dir not found: {model_dir}")
    if not os.path.isdir(processor_dir):
        raise FileNotFoundError(f"--processor_dir not found: {processor_dir}")
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"--jsonl not found: {jsonl_path}")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"--base_dir not found: {base_dir}")
    if not os.path.isfile(codebook_json):
        raise FileNotFoundError(f"--codebook_json not found: {codebook_json}")

    # Load processor/tokenizer
    processor = AutoProcessor.from_pretrained(
        processor_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = processor.tokenizer

    # dtype & device
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if use_cuda:
        dtype = torch.float16 if args.force_fp16 else torch.bfloat16
    else:
        dtype = torch.float32

    # Load model on SINGLE GPU for this rank
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    # vocab size safety (same as training)
    if getattr(model.config, "vocab_size", None) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    # Load codebook
    codebook = Codebook5x3(codebook_json)

    # Load dataset offsets (each rank loads offsets; cheap)
    ds = JsonlOffsetDataset(
        jsonl_path=jsonl_path,
        token_ctx_key=args.token_ctx_key,
        require_prompt=args.require_prompt,
        past_len=args.past_len,
        future_len=args.future_len,
    )

    # Determine indices to process
    start = max(0, int(args.start_idx))
    end = min(len(ds), start + int(args.num_samples))
    all_indices = list(range(start, end))
    my_indices = all_indices[rank::world]

    # result log per rank
    res_path = os.path.join(out_rank_dir, "results.jsonl")
    fo = open(res_path, "w", encoding="utf-8")

    # progress bar only on rank0 (others silent to avoid messy logs)
    it = tqdm(my_indices, desc=f"rank{rank} infer", disable=(rank != 0))

    for j, idx in enumerate(it):
        ex = ds[idx]
        paths = ex.get("paths", {})
        if not isinstance(paths, dict):
            continue

        # images
        p_left = _join_base(base_dir, paths.get(args.img_left_key))
        p_mid = _join_base(base_dir, paths.get(args.img_mid_key))
        p_right = _join_base(base_dir, paths.get(args.img_right_key))
        if (p_left is None) or (p_mid is None) or (p_right is None):
            continue
        if not (os.path.exists(p_left) and os.path.exists(p_mid) and os.path.exists(p_right)):
            continue

        img_left = open_rgb(p_left)
        img_mid = open_rgb(p_mid)
        img_right = open_rgb(p_right)
        images = [img_mid, img_left, img_right]  # order must match training: middle, left, right

        # build prompt text (same as training prompt side)
        try:
            user_text, past_tokens = build_user_text(
                ex=ex,
                token_ctx_key=args.token_ctx_key,
                prefix=args.prefix,
                past_len=args.past_len,
            )
        except Exception:
            continue

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
        prompt_text = processor.apply_chat_template(
            mm_msgs_prompt, tokenize=False, add_generation_prompt=True
        )

        # encode inputs
        inputs = processor(
            text=[prompt_text],
            images=[[img_mid, img_left, img_right]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )

        # move to device
        for k in list(inputs.keys()):
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].to(device)

        # generate
        gen_kwargs = dict(
            max_new_tokens=int(args.max_new_tokens),
            do_sample=bool(args.do_sample),
        )
        if args.do_sample:
            gen_kwargs.update(dict(
                temperature=float(args.temperature),
                top_p=float(args.top_p),
            ))

        with torch.inference_mode():
            out_ids = model.generate(**inputs, **gen_kwargs)

        # decode and parse tokens
        gen_text = tokenizer.decode(out_ids[0], skip_special_tokens=False)
        pred_tokens = parse_traj_tokens_from_text(gen_text, prefix=args.prefix)

        # keep only first future_len tokens
        if len(pred_tokens) >= args.future_len:
            pred_tokens = pred_tokens[:args.future_len]
        else:
            # pad with zeros if model outputs fewer tokens
            pred_tokens = pred_tokens + [0] * (args.future_len - len(pred_tokens))

        # GT tokens
        ctx = ex.get(args.token_ctx_key, {})
        gt_future = ctx.get("tokens_future_5s", [])
        if not isinstance(gt_future, list) or len(gt_future) != args.future_len:
            continue
        gt_future = [int(x) for x in gt_future]

        # reconstruct XY
        past_steps_xy = codebook.tokens_to_steps_xy(past_tokens)                 # [30,2]
        gt_steps_xy = codebook.tokens_to_steps_xy(gt_future)                    # [50,2]
        pred_steps_xy = codebook.tokens_to_steps_xy(pred_tokens)                # [50,2]

        past_xy = steps_to_xy(past_steps_xy, start_at_origin=True, end_at_origin=True)   # ends at (0,0)
        gt_xy = steps_to_xy(gt_steps_xy, start_at_origin=True, end_at_origin=False)      # starts at (0,0)
        pred_xy = steps_to_xy(pred_steps_xy, start_at_origin=True, end_at_origin=False)  # starts at (0,0)

        # plot
        time_str = ex.get("time_str", f"idx{idx}")
        title = f"{time_str} | idx={idx} | rank={rank}"
        out_png = os.path.join(out_rank_dir, f"{idx:07d}_{time_str}_xy.png")
        plot_xy_compare(out_png, past_xy, gt_xy, pred_xy, title=title)

        # save a small jsonl record
        rec_out = {
            "idx": int(idx),
            "time_str": time_str,
            "rank": int(rank),
            "prompt": ex.get("prompt", ""),
            "past_tokens": past_tokens,
            "gt_future_tokens": gt_future,
            "pred_future_tokens": pred_tokens,
            "png": out_png,
        }
        fo.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

        # optional cache cleaning
        if args.empty_cache_every > 0 and (j % int(args.empty_cache_every) == 0):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    fo.close()

    ddp_barrier()

    # Merge summaries on rank0 (optional, lightweight)
    if rank == 0:
        merged_path = os.path.join(out_dir, "results_merged.jsonl")
        with open(merged_path, "w", encoding="utf-8") as fout:
            for r in range(world):
                rp = os.path.join(out_dir, f"rank{r}", "results.jsonl")
                if not os.path.exists(rp):
                    continue
                with open(rp, "r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
        print(f"✅ merged results -> {merged_path}")
        print(f"✅ vis dirs -> {out_dir}/rank*/")

    ddp_destroy()


if __name__ == "__main__":
    main()

