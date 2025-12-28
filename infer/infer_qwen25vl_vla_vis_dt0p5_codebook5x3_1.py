#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_qwen25vl_vla_vis_dt0p5_codebook5x3.py

Inference + visualization for Qwen2.5-VL VLA trajectory tokens.
- 3 images: middle (zed_left), left (front_left), right (front_right)
- Input includes prompt + (optional) vehicle state + past tokens
- Model outputs 10 future tokens: <traj_0000> ... <traj_2047>
- Decode tokens via KMeans codebook json:
    each token -> 15D = 5*(dx,dy,dphi) at 10Hz (0.1s)
    10 tokens -> 50 steps -> 5.0s

Visualization:
- Past trajectory (3.0s, 30 steps) reconstructed from tokens_past_3s
- GT future trajectory (5.0s, 50 steps) reconstructed from tokens_future_5s (quantized GT)
- Pred future trajectory reconstructed from predicted tokens
- Save per-sample PNG and a summary JSON with tokens + ADE/FDE

Example:
CUDA_VISIBLE_DEVICES=0,1 python infer_qwen25vl_vla_vis_dt0p5_codebook5x3.py \
  --model_dir "/home/lxh/traj-vla/12.27-auto-vla/ckpt_grouped_full" \
  --processor_dir "/home/lxh/traj-vla/12.27-auto-vla/ckpt_grouped_full" \
  --jsonl "/home/lxh/traj-vla/12.27-auto-vla/records_ctx_tokens_only_past3s_future5s_dt0p5.jsonl" \
  --base_dir "/home/lxh/lxh/vla_data" \
  --codebook_json "/home/lxh/traj-vla/12.27-auto-vla/action_codebook_future5frames_kmeans2048_dt0p5.json" \
  --out_dir "/home/lxh/traj-vla/12.27-auto-vla/infer_vis" \
  --num_samples 50 --start_idx 0 \
  --frame_mode per_step

Notes on frame_mode:
- per_step   : treat each (dx,dy) as local in current yaw, rotate and integrate (matches your verbal definition)
- fixed_anchor: treat (dx,dy) already in ego@t0 frame, no rotation (matches your token building script's anchor_i=i usage)
"""

import os
import re
import json
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from PIL import Image, ImageOps

# -------------------------
# Early parse: set CUDA_VISIBLE_DEVICES BEFORE torch import
# -------------------------
def _early_parse_cuda_visible_devices() -> str:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--cuda_visible_devices", type=str, default="0,1,2,3",
                    help='Comma-separated GPU ids, e.g. "0,1".')
    args, _ = ap.parse_known_args()
    return args.cuda_visible_devices

os.environ["CUDA_VISIBLE_DEVICES"] = _early_parse_cuda_visible_devices()

# Now import torch/transformers
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


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

def open_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


# -------------------------
# token/text helpers
# -------------------------
def tokens_to_traj_str(tokens: List[int], prefix: str = "traj") -> str:
    return " ".join([f"<{prefix}_{t:04d}>" for t in tokens])

def parse_traj_tokens_from_text(text: str, prefix: str = "traj") -> List[int]:
    # capture <traj_0000> like
    pat = re.compile(rf"<{re.escape(prefix)}_(\d{{4}})>")
    ids = []
    for m in pat.finditer(text):
        try:
            ids.append(int(m.group(1)))
        except Exception:
            pass
    return ids


# -------------------------
# state extract (best-effort) — consistent with your trainer
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
# Codebook loader (JSON)
# -------------------------
class CodebookKMeansDecoder:
    """
    Decode token -> 5x3 (dx,dy,dphi) at 10Hz (0.1s).
    We use centers_unscaled_for_reconstruction as unscaled real increments.
    """
    def __init__(self, cb_json_path: str):
        p = Path(cb_json_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(str(p))
        with p.open("r", encoding="utf-8") as f:
            cb = json.load(f)

        if "centers_unscaled_for_reconstruction" not in cb:
            raise KeyError("codebook missing centers_unscaled_for_reconstruction")

        self.name = cb.get("name", p.name)
        self.centers = np.asarray(cb["centers_unscaled_for_reconstruction"], dtype=np.float32)  # [K,15]
        self.K = int(cb.get("k", self.centers.shape[0]))
        self.chunk_steps = int(cb.get("chunk_steps", 5))  # should be 5
        self.token_dim = int(cb.get("token_dim", 15))     # should be 15
        self.dt_token = float(cb.get("dt_token", cb.get("dt", 0.5)))
        self.hz_src = float(cb.get("hz_src", 10.0))

        if self.centers.ndim != 2 or self.centers.shape[1] != self.token_dim:
            raise ValueError(f"centers shape mismatch: {self.centers.shape} token_dim={self.token_dim}")

        if self.token_dim != 15 or self.chunk_steps != 5:
            raise ValueError(f"This script expects token_dim=15 and chunk_steps=5, got {self.token_dim},{self.chunk_steps}")

    def decode_token(self, tid: int) -> np.ndarray:
        if tid < 0:
            tid = 0
        if tid >= self.centers.shape[0]:
            tid = int(np.clip(tid, 0, self.centers.shape[0]-1))
        v15 = self.centers[tid]  # [15]
        return v15.reshape(5, 3)  # [5,3] (dx,dy,dphi)


# -------------------------
# Integrate dx,dy,dphi -> XY trajectory
# -------------------------
def integrate_steps(
    steps_5x3_list: List[np.ndarray],
    frame_mode: str = "per_step",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    steps_5x3_list: list of [5,3] arrays, total steps = 5*len(list)
    return:
      xy: [T+1,2] positions with xy[0]=(0,0)
      yaw: [T+1] yaw with yaw[0]=0
    frame_mode:
      - per_step: rotate each (dx,dy) by current yaw before accumulating
      - fixed_anchor: directly sum dx,dy without rotation
    """
    steps = np.concatenate(steps_5x3_list, axis=0).astype(np.float32)  # [T,3]
    T = steps.shape[0]
    xy = np.zeros((T + 1, 2), dtype=np.float32)
    yaw = np.zeros((T + 1,), dtype=np.float32)

    for t in range(T):
        dx, dy, dphi = float(steps[t, 0]), float(steps[t, 1]), float(steps[t, 2])
        cur_yaw = float(yaw[t])

        if frame_mode == "per_step":
            # rotate local (dx,dy) into initial frame by cur_yaw
            c = math.cos(cur_yaw)
            s = math.sin(cur_yaw)
            dx_g = c * dx - s * dy
            dy_g = s * dx + c * dy
            xy[t + 1, 0] = xy[t, 0] + dx_g
            xy[t + 1, 1] = xy[t, 1] + dy_g
        elif frame_mode == "fixed_anchor":
            xy[t + 1, 0] = xy[t, 0] + dx
            xy[t + 1, 1] = xy[t, 1] + dy
        else:
            raise ValueError(f"Unknown frame_mode: {frame_mode}")

        yaw[t + 1] = yaw[t] + dphi

    return xy, yaw


def build_prompt_and_images(
    ex: Dict[str, Any],
    processor: Any,
    base_dir: str,
    token_ctx_key: str,
    prefix: str,
    img_left_key: str,
    img_mid_key: str,
    img_right_key: str,
    past_len: int,
) -> Tuple[str, List[Image.Image], List[int]]:
    """
    Returns:
      prompt_text (chat template with generation prompt)
      images ordered [mid, left, right]
      past_tokens list[int]
    """
    paths = ex.get("paths", {})
    if not isinstance(paths, dict):
        raise KeyError("paths must be dict")

    p_left = _join_base(base_dir, paths.get(img_left_key))
    p_mid = _join_base(base_dir, paths.get(img_mid_key))
    p_right = _join_base(base_dir, paths.get(img_right_key))
    for p in [p_left, p_mid, p_right]:
        if (p is None) or (not os.path.exists(p)):
            raise FileNotFoundError(f"Image not found: {p}")

    img_left = open_rgb(p_left)
    img_mid = open_rgb(p_mid)
    img_right = open_rgb(p_right)
    images = [img_mid, img_left, img_right]

    prompt = ex.get("prompt", "")
    if not isinstance(prompt, str):
        prompt = str(prompt)

    v = get_speed_mps(ex)
    steer = get_steer_deg(ex)
    yaw_deg = get_yaw_deg(ex)

    state_parts = []
    if v is not None:
        state_parts.append(f"speed={v:.3f}")
    if steer is not None:
        state_parts.append(f"steer_deg={steer:.3f}")
    if yaw_deg is not None:
        state_parts.append(f"heading_deg={yaw_deg:.3f}")
    state_str = ", ".join(state_parts) if state_parts else "N/A"

    ctx = ex.get(token_ctx_key, {})
    tp = ctx.get("tokens_past_3s", [])
    if not isinstance(tp, list) or len(tp) != past_len:
        raise ValueError(f"Bad past tokens in {token_ctx_key}: got {type(tp)} len={len(tp) if isinstance(tp, list) else 'NA'}")
    tp = [int(x) for x in tp]

    past_str = tokens_to_traj_str(tp, prefix=prefix)

    user_text = (
        "You are an autonomous driving agent.\n"
        "You are given THREE camera images: middle, left, right.\n"
        f"Driving prompt (intent): {prompt}\n"
        f"Vehicle state: {state_str}\n"
        f"Past trajectory tokens (past 3.0s, dt=0.5s, 6 tokens): {past_str}\n\n"
        "Task: Predict the future trajectory tokens for the next 5.0s (dt=0.5s, 10 tokens).\n"
        "Output ONLY the 10 tokens, separated by spaces. Do not output any other words.\n"
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
    prompt_text = processor.apply_chat_template(
        mm_msgs_prompt, tokenize=False, add_generation_prompt=True
    )

    return prompt_text, images, tp


def get_future_tokens_gt(ex: Dict[str, Any], token_ctx_key: str, future_len: int) -> List[int]:
    ctx = ex.get(token_ctx_key, {})
    tf = ctx.get("tokens_future_5s", [])
    if not isinstance(tf, list) or len(tf) != future_len:
        raise ValueError(f"Bad future tokens in {token_ctx_key}: got {type(tf)} len={len(tf) if isinstance(tf, list) else 'NA'}")
    return [int(x) for x in tf]


def compute_ade_fde(pred_xy: np.ndarray, gt_xy: np.ndarray) -> Tuple[float, float]:
    """
    pred_xy, gt_xy: [T+1,2], include origin
    ADE: mean distance over future points (exclude t=0)
    FDE: distance at final point
    """
    T = min(pred_xy.shape[0], gt_xy.shape[0])
    if T <= 1:
        return float("nan"), float("nan")
    p = pred_xy[:T]
    g = gt_xy[:T]
    d = np.linalg.norm(p[1:] - g[1:], axis=1)
    ade = float(np.mean(d))
    fde = float(np.linalg.norm(p[T-1] - g[T-1]))
    return ade, fde


def save_plot(
    out_png: str,
    past_xy: np.ndarray,
    gt_future_xy: np.ndarray,
    pred_future_xy: np.ndarray,
    title: str,
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 7))
    # past ends at origin
    plt.plot(past_xy[:, 0], past_xy[:, 1], label="Past (3s)", linewidth=2)
    plt.plot(gt_future_xy[:, 0], gt_future_xy[:, 1], label="GT future (tokens)", linewidth=2)
    plt.plot(pred_future_xy[:, 0], pred_future_xy[:, 1], label="Pred future", linewidth=2)

    plt.scatter([0.0], [0.0], s=50, marker="x", label="t=0")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.xlabel("x (m, forward)")
    plt.ylabel("y (m, left)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda_visible_devices", type=str, default="0,1")
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

    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--num_samples", type=int, default=50)
    ap.add_argument("--start_idx", type=int, default=0)

    ap.add_argument("--frame_mode", type=str, default="per_step",
                    choices=["per_step", "fixed_anchor"],
                    help="How to integrate dx,dy: per_step rotates by yaw; fixed_anchor sums directly.")

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    print(f"[CUDA_VISIBLE_DEVICES] {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    if torch.cuda.is_available():
        print(f"[CUDA] visible device count = {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - cuda:{i} = {torch.cuda.get_device_name(i)}")

    model_dir = str(Path(args.model_dir).expanduser().resolve())
    processor_dir = str(Path(args.processor_dir).expanduser().resolve()) if args.processor_dir else model_dir
    jsonl_path = str(Path(args.jsonl).expanduser().resolve())
    base_dir = str(Path(args.base_dir).expanduser().resolve())
    codebook_json = str(Path(args.codebook_json).expanduser().resolve())
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # load codebook
    codebook = CodebookKMeansDecoder(codebook_json)
    print(f"[Codebook] {codebook.name} K={codebook.K} token_dim={codebook.token_dim} chunk_steps={codebook.chunk_steps}")

    # load model / processor
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    processor = AutoProcessor.from_pretrained(
        processor_dir, trust_remote_code=True, local_files_only=True
    )
    tokenizer = processor.tokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    model.config.use_cache = True

    # ensure vocab match
    if getattr(model.config, "vocab_size", None) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    # dataset
    ds = JsonlOffsetDataset(
        jsonl_path=jsonl_path,
        token_ctx_key=args.token_ctx_key,
        require_prompt=True,   # 你训练用了 --require_prompt
        past_len=args.past_len,
        future_len=args.future_len,
    )

    start = int(args.start_idx)
    end = min(len(ds), start + int(args.num_samples))
    if start < 0 or start >= len(ds):
        raise ValueError(f"start_idx out of range: {start} / {len(ds)}")

    # pick first device for inputs
    first_device = next(iter(model.parameters())).device
    print(f"[Model] first_device = {first_device}")

    results: List[Dict[str, Any]] = []

    for idx in tqdm(range(start, end), desc="Infer"):
        ex = ds[idx]

        # build prompt and images
        try:
            prompt_text, images, past_tokens = build_prompt_and_images(
                ex=ex,
                processor=processor,
                base_dir=base_dir,
                token_ctx_key=args.token_ctx_key,
                prefix=args.prefix,
                img_left_key=args.img_left_key,
                img_mid_key=args.img_mid_key,
                img_right_key=args.img_right_key,
                past_len=args.past_len,
            )
            gt_future_tokens = get_future_tokens_gt(ex, args.token_ctx_key, args.future_len)
        except Exception as e:
            results.append({"idx": idx, "error": f"build_input_fail: {repr(e)}"})
            continue

        # encode
        # NOTE: For Qwen2.5-VL, pass text + images; images must be nested list: [[mid,left,right]]
        model_inputs = processor(
            text=[prompt_text],
            images=[images],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )

        # move to first_device (important for sharded models)
        for k in list(model_inputs.keys()):
            if isinstance(model_inputs[k], torch.Tensor):
                model_inputs[k] = model_inputs[k].to(first_device)

        gen_kwargs = dict(
            max_new_tokens=int(args.max_new_tokens),
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        with torch.inference_mode():
            out_ids = model.generate(**model_inputs, **gen_kwargs)

        # decode text
        gen_text = tokenizer.decode(out_ids[0], skip_special_tokens=False)

        # parse traj tokens from full generated text
        pred_ids_all = parse_traj_tokens_from_text(gen_text, prefix=args.prefix)

        # Heuristic: take the last future_len tokens (often prompt contains past tokens too)
        if len(pred_ids_all) >= args.future_len:
            pred_future_tokens = pred_ids_all[-args.future_len:]
        else:
            # fallback: pad with zeros
            pred_future_tokens = pred_ids_all + [0] * (args.future_len - len(pred_ids_all))

        pred_future_tokens = [int(np.clip(t, 0, codebook.K - 1)) for t in pred_future_tokens]

        # decode trajectories
        # past: 6 tokens -> 30 steps -> make it end at origin
        past_steps_list = [codebook.decode_token(tid) for tid in past_tokens]      # list of [5,3]
        past_xy_raw, past_yaw_raw = integrate_steps(past_steps_list, frame_mode=args.frame_mode)
        # shift so that t=0 at origin (end point is origin)
        past_xy = past_xy_raw - past_xy_raw[-1:,:]

        # gt future (quantized tokens)
        gt_steps_list = [codebook.decode_token(tid) for tid in gt_future_tokens]
        gt_future_xy, gt_future_yaw = integrate_steps(gt_steps_list, frame_mode=args.frame_mode)

        # pred future
        pred_steps_list = [codebook.decode_token(tid) for tid in pred_future_tokens]
        pred_future_xy, pred_future_yaw = integrate_steps(pred_steps_list, frame_mode=args.frame_mode)

        ade, fde = compute_ade_fde(pred_future_xy, gt_future_xy)

        # title
        time_str = ex.get("time_str", "")
        prompt = ex.get("prompt", "")
        title = f"idx={idx}  time={time_str}  prompt={prompt}\nframe_mode={args.frame_mode}  ADE={ade:.3f}m  FDE={fde:.3f}m"

        # save fig
        out_png = out_dir / f"vis_{idx:06d}.png"
        try:
            save_plot(str(out_png), past_xy, gt_future_xy, pred_future_xy, title=title)
        except Exception as e:
            results.append({"idx": idx, "error": f"plot_fail: {repr(e)}"})
            continue

        results.append({
            "idx": idx,
            "time_str": time_str,
            "prompt": prompt,
            "frame_mode": args.frame_mode,
            "past_tokens": past_tokens,
            "gt_future_tokens": gt_future_tokens,
            "pred_future_tokens": pred_future_tokens,
            "ade_m": ade,
            "fde_m": fde,
            "png": str(out_png),
            "gen_text_tail": gen_text[-400:],  # debug: last 400 chars
        })

    # save summary
    out_json = out_dir / "infer_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({
            "model_dir": model_dir,
            "processor_dir": processor_dir,
            "jsonl": jsonl_path,
            "base_dir": base_dir,
            "codebook_json": codebook_json,
            "token_ctx_key": args.token_ctx_key,
            "prefix": args.prefix,
            "start_idx": start,
            "end_idx": end,
            "num_samples": args.num_samples,
            "frame_mode": args.frame_mode,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. Saved figures to: {out_dir}")
    print(f"✅ Summary json: {out_json}")


if __name__ == "__main__":
    main()
