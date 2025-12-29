#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_plot_traj_xy_kmeans_chunk.py

- Qwen2.5-VL base + LoRA adapter inference
- Decode traj tokens with your kmeans codebook json:
  centers_unscaled_for_reconstruction: (K, token_dim=15)
  chunk_steps=5, dims: [dx,dy,dphi] repeated 5 times
- Reconstruct XY for past / future GT / future Pred and plot in one figure

Example:
python infer_plot_traj_xy_kmeans_chunk.py \
  --model_dir "/home/lxh/lxh/12.26/qwen2_5vl_with_traj_tokens_2048" \
  --lora_dir  "/home/lxh/traj-vla/ckpt_lora_1img_ddp_dropstraight" \
  --processor_dir "/home/lxh/lxh/12.26/qwen2_5vl_with_traj_tokens_2048" \
  --jsonl "/home/lxh/traj-vla/12.27-auto-vla/records_ctx_tokens_only_past3s_future5s_dt0p5.jsonl" \
  --base_dir "/home/lxh/lxh/vla_data" \
  --codebook "/path/to/action_codebook_future5frames_kmeans2048_dt0p5.json" \
  --token_ctx_key "token_ctx_dt0p5_past3s_future5s" \
  --img_main_key "zed_left" \
  --idx 0 \
  --out_png "viz_xy.png"
"""

import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel


# -------------------------
# basic utils
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

def tokens_to_traj_str(tokens: List[int], prefix: str = "traj") -> str:
    return " ".join([f"<{prefix}_{int(t):04d}>" for t in tokens])

def parse_traj_tokens_from_text(text: str, prefix: str = "traj") -> List[int]:
    # Extract <traj_0001> tokens in order
    pat = re.compile(rf"<{re.escape(prefix)}_(\d{{4}})>")
    return [int(m.group(1)) for m in pat.finditer(text)]

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
# codebook (your json)
# -------------------------
def load_kmeans_chunk_codebook_json(path: str) -> Tuple[np.ndarray, int, float]:
    """
    Read your codebook json and return:
      steps: (K, chunk_steps, 3) float32 => [dx, dy, dphi] for each substep
      chunk_steps: int
      dt_token: float
    Uses: centers_unscaled_for_reconstruction
    """
    path = str(Path(path).expanduser().resolve())
    if not os.path.isfile(path):
        raise FileNotFoundError(f"codebook not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if "centers_unscaled_for_reconstruction" not in obj:
        raise KeyError("codebook json missing key: centers_unscaled_for_reconstruction")

    centers = np.asarray(obj["centers_unscaled_for_reconstruction"], dtype=np.float32)  # (K, token_dim)
    if centers.ndim != 2:
        raise ValueError(f"Bad centers shape: {centers.shape}")

    chunk_steps = int(obj.get("chunk_steps", 5))
    dt_token = float(obj.get("dt_token", 0.5))

    token_dim = centers.shape[1]
    if token_dim != chunk_steps * 3:
        raise ValueError(f"token_dim mismatch: token_dim={token_dim} but chunk_steps*3={chunk_steps*3}")

    steps = centers.reshape(centers.shape[0], chunk_steps, 3)  # (K,5,3)
    return steps, chunk_steps, dt_token


# -------------------------
# integrate tokens -> XY
# -------------------------
def integrate_tokens_to_xy(
    tokens: List[int],
    code_steps: np.ndarray,    # (K, chunk_steps, 3)
    x0: float = 0.0,
    y0: float = 0.0,
    yaw0: float = 0.0,
    record: str = "chunk",     # "chunk" => only token endpoints; "step" => 0.1s substeps points
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Each token -> chunk_steps substeps [dx,dy,dphi] in ego frame.
    Rotate (dx,dy) by current yaw to world, then accumulate.

    record:
      - "chunk": return len(tokens)+1 points (token endpoints)
      - "step":  return 1 + len(tokens)*chunk_steps points (substep endpoints)
    """
    xs = [float(x0)]
    ys = [float(y0)]
    yaws = [float(yaw0)]

    K = code_steps.shape[0]
    chunk_steps = code_steps.shape[1]

    for t in tokens:
        t = int(t)
        if t < 0 or t >= K:
            raise ValueError(f"Token out of range: {t} (K={K})")

        # apply substeps
        for s in range(chunk_steps):
            dx, dy, dphi = code_steps[t, s].tolist()
            yaw = yaws[-1]

            c = math.cos(yaw)
            s_ = math.sin(yaw)
            wx = c * dx - s_ * dy
            wy = s_ * dx + c * dy

            xs.append(xs[-1] + wx)
            ys.append(ys[-1] + wy)
            yaws.append(yaw + dphi)

        if record == "chunk":
            # keep only chunk endpoint: drop intermediate points just added,
            # but keep last one (endpoint)
            # We can compress by slicing after loop; easiest: do nothing here and compress at end.
            pass

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    yaws = np.asarray(yaws, dtype=np.float32)

    if record == "chunk":
        # pick indices: 0, chunk_steps, 2*chunk_steps, ...
        idxs = [0] + [(i * chunk_steps) for i in range(1, len(tokens) + 1)]
        xs = xs[idxs]
        ys = ys[idxs]
        yaws = yaws[idxs]

    return xs, ys, yaws


# -------------------------
# jsonl loader
# -------------------------
def load_sample(jsonl_path: str, idx: Optional[int], sample_id: Optional[str]) -> Dict[str, Any]:
    jsonl_path = str(Path(jsonl_path).expanduser().resolve())
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(jsonl_path)

    if idx is None and (not sample_id):
        idx = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        if sample_id:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if str(obj.get("id", "")) == str(sample_id):
                    return obj
            raise KeyError(f"sample id not found: {sample_id}")
        else:
            j = -1
            for line in f:
                if not line.strip():
                    continue
                j += 1
                if j == idx:
                    return json.loads(line)
            raise IndexError(f"idx out of range: {idx}")


def build_user_text(
    ex: Dict[str, Any],
    token_ctx_key: str,
    prefix: str,
    past_len: int,
) -> Tuple[str, List[int], List[int], str]:
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
    tf = ctx.get("tokens_future_5s", [])

    if not isinstance(tp, list) or not isinstance(tf, list):
        raise ValueError("tokens_past_3s/tokens_future_5s must be list")
    if len(tp) != past_len:
        raise ValueError(f"past token length mismatch: expect {past_len}, got {len(tp)}")

    past_tokens = [int(x) for x in tp]
    future_tokens = [int(x) for x in tf]

    past_str = tokens_to_traj_str(past_tokens, prefix=prefix)

    user_text = (
        "You are an autonomous driving agent.\n"
        "You are given ONE forward main-view camera image.\n"
        f"Driving prompt (intent): {prompt}\n"
        f"Vehicle state: {state_str}\n"
        f"Past trajectory tokens (past 3.0s, dt=0.5s, {past_len} tokens): {past_str}\n\n"
        "Task: Predict the future trajectory tokens for the next 5.0s (dt=0.5s, 10 tokens).\n"
        "Output ONLY the 10 tokens, separated by spaces. Do not output any other words.\n"
    )
    return user_text, past_tokens, future_tokens, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--processor_dir", default=None)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--base_dir", required=True)

    ap.add_argument("--codebook", required=True, help="action_codebook_future5frames_kmeans2048_dt0p5.json")
    ap.add_argument("--token_ctx_key", default="token_ctx_dt0p5_past3s_future5s")
    ap.add_argument("--img_main_key", default="zed_left")
    ap.add_argument("--prefix", default="traj")
    ap.add_argument("--past_len", type=int, default=6)
    ap.add_argument("--future_len", type=int, default=10)

    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--id", type=str, default=None)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--record", type=str, default="chunk", choices=["chunk", "step"],
                    help="chunk: plot each token endpoint; step: plot 0.1s substeps within each token")

    ap.add_argument("--out_png", type=str, default="traj_xy_compare.png")
    ap.add_argument("--title", type=str, default=None)

    args = ap.parse_args()

    model_dir = str(Path(args.model_dir).expanduser().resolve())
    lora_dir = str(Path(args.lora_dir).expanduser().resolve())
    processor_dir = str(Path(args.processor_dir).expanduser().resolve()) if args.processor_dir else model_dir
    jsonl_path = str(Path(args.jsonl).expanduser().resolve())
    base_dir = str(Path(args.base_dir).expanduser().resolve())

    # load codebook
    code_steps, chunk_steps, dt_token = load_kmeans_chunk_codebook_json(args.codebook)
    # dt per substep:
    dt_step = dt_token / float(chunk_steps)

    # load one sample
    ex = load_sample(jsonl_path, args.idx, args.id)
    paths = ex.get("paths", {})
    if not isinstance(paths, dict):
        raise KeyError("sample['paths'] must be dict")
    img_path = _join_base(base_dir, paths.get(args.img_main_key))
    if not img_path or (not os.path.isfile(img_path)):
        raise FileNotFoundError(f"Main image not found: {img_path} (key={args.img_main_key})")

    user_text, past_tokens, future_tokens_gt, prompt_str = build_user_text(
        ex, token_ctx_key=args.token_ctx_key, prefix=args.prefix, past_len=args.past_len
    )

    # load processor & model
    processor = AutoProcessor.from_pretrained(processor_dir, trust_remote_code=True, local_files_only=True)
    tokenizer = processor.tokenizer

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda") else torch.float32

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True, torch_dtype=dtype, device_map=None
    )
    if getattr(base_model.config, "vocab_size", None) != len(tokenizer):
        base_model.resize_token_embeddings(len(tokenizer))
        base_model.config.vocab_size = len(tokenizer)
        if hasattr(base_model, "tie_weights"):
            base_model.tie_weights()

    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()
    model.to(device)

    # build inputs
    img = open_rgb(img_path)
    mm_msgs = [
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": user_text}]}
    ]
    prompt_text = processor.apply_chat_template(mm_msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt_text],
        images=[[img]],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # generate
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
        )

    gen_text = tokenizer.decode(out_ids[0], skip_special_tokens=False)
    pred_tokens_all = parse_traj_tokens_from_text(gen_text, prefix=args.prefix)
    if len(pred_tokens_all) >= args.future_len:
        pred_tokens = pred_tokens_all[-args.future_len:]
    else:
        pred_tokens = pred_tokens_all

    if len(pred_tokens) != args.future_len:
        print("[Warn] predicted token count != future_len")
        print("  pred_tokens_all:", pred_tokens_all)
        print("  used_pred_tokens:", pred_tokens)

    # reconstruct XY:
    # past from origin
    x_p, y_p, yaw_p = integrate_tokens_to_xy(past_tokens, code_steps, 0.0, 0.0, 0.0, record=args.record)
    x0, y0, yaw0 = float(x_p[-1]), float(y_p[-1]), float(yaw_p[-1])

    # future start from past end
    x_gt, y_gt, _ = integrate_tokens_to_xy(future_tokens_gt, code_steps, x0, y0, yaw0, record=args.record)
    x_pr, y_pr, _ = integrate_tokens_to_xy(pred_tokens, code_steps, x0, y0, yaw0, record=args.record)

    # plot XY
    plt.figure(figsize=(7, 7))
    plt.plot(x_p, y_p, marker="o", linewidth=2, label=f"Past (recon, dt={dt_step:.2f}s)" if args.record=="step" else "Past (recon)")
    plt.plot(x_gt, y_gt, marker="o", linewidth=2, label="Future GT (recon)")
    plt.plot(x_pr, y_pr, marker="o", linewidth=2, label="Future Pred (recon)")
    plt.scatter([x0], [y0], s=80, marker="x", label="t=0 (past end)")

    plt.axis("equal")
    plt.grid(True, linestyle="--", linewidth=0.8)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    sid = ex.get("id", None)
    title = args.title or f"XY: Past + Future (GT vs Pred)\nID={sid} | prompt={prompt_str} | token=0.5s, step={dt_step:.2f}s"
    plt.title(title)
    plt.legend()

    out_png = str(Path(args.out_png).expanduser().resolve())
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

    print("✅ saved:", out_png)
    print("codebook:", str(Path(args.codebook).expanduser().resolve()))
    print("chunk_steps:", chunk_steps, "dt_token:", dt_token, "dt_step:", dt_step)
    print("pred_tokens:", pred_tokens)


if __name__ == "__main__":
    main()
