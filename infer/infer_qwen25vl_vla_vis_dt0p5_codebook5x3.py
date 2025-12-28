#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Infer Qwen2.5-VL VLA (3 images + prompt/state + past tokens -> future tokens)
and visualize XY trajectories:
- history trajectory (past 3.0s, dt=0.1s reconstructed from 6 tokens * (5 points))
- ground-truth future trajectory (next 5.0s, dt=0.1s reconstructed from 10 tokens)
- predicted future trajectory (next 5.0s, dt=0.1s reconstructed from predicted 10 tokens)

Your token codebook:
- Each token represents 0.5s chunk and contains 5 points at 10Hz.
- codebook is (K, 5, 3) or (K, 15) where each point is [dx, dy, dphi] (or any 3 dims; we use dx,dy always).
"""

import os
import re
import gc
import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------
# Early parse: set CUDA_VISIBLE_DEVICES BEFORE torch import
# -------------------------
def _early_parse_cuda_visible_devices() -> str:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--cuda_visible_devices", type=str, default="0,1",
                    help='Comma-separated GPU ids, e.g. "0" or "0,1".')
    args, _ = ap.parse_known_args()
    return args.cuda_visible_devices

os.environ["CUDA_VISIBLE_DEVICES"] = _early_parse_cuda_visible_devices()

import torch
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
    pat = re.compile(rf"<{re.escape(prefix)}_(\d{{4}})>")
    out = []
    for m in pat.finditer(text):
        try:
            out.append(int(m.group(1)))
        except Exception:
            pass
    return out


# -------------------------
# state extract (best-effort) - same logic as your training
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
# JSONL offsets (fast random access)
# -------------------------
class JsonlOffsetIndex:
    def __init__(self, jsonl_path: str):
        self.jsonl_path = str(Path(jsonl_path).expanduser().resolve())
        if not os.path.isfile(self.jsonl_path):
            raise FileNotFoundError(self.jsonl_path)
        self.offsets: List[int] = []
        with open(self.jsonl_path, "rb") as f:
            off = 0
            for line in tqdm(f, desc="Index jsonl"):
                self.offsets.append(off)
                off += len(line)

    def __len__(self):
        return len(self.offsets)

    def get(self, idx: int) -> Dict[str, Any]:
        off = self.offsets[idx]
        with open(self.jsonl_path, "rb") as f:
            f.seek(off)
            line = f.readline()
        return json.loads(line.decode("utf-8"))


# -------------------------
# codebook: token -> (5,3) increments
# -------------------------
def load_codebook(codebook_path: str, K: int) -> np.ndarray:
    """
    Expected:
      - (K,5,3)  or
      - (K,15) -> reshape to (K,5,3)
    """
    p = str(Path(codebook_path).expanduser().resolve())
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    cb = np.load(p)
    if cb.shape[0] != K:
        raise ValueError(f"Codebook K mismatch: got {cb.shape[0]}, expected {K}. shape={cb.shape}")
    if cb.ndim == 3 and cb.shape[1] == 5 and cb.shape[2] >= 3:
        return cb[:, :, :3].astype(np.float32)
    if cb.ndim == 2 and cb.shape[1] == 15:
        cb = cb.reshape(K, 5, 3).astype(np.float32)
        return cb
    raise ValueError(f"Unsupported codebook shape: {cb.shape}. Need (K,5,3) or (K,15).")


def integrate_token_sequence_to_xy_10hz(
    token_ids: List[int],
    codebook: np.ndarray,
    rotate_by_yaw: bool = False,
) -> np.ndarray:
    """
    token_ids length: N tokens
    codebook[token] -> (5,3): 5 points at 10Hz in this 0.5s chunk
      each row: [dx, dy, dphi] (we use dx,dy always; dphi optional)

    Return: positions (T,2) at 10Hz, starting from (0,0) not included.
      T = N*5
    """
    xs, ys = 0.0, 0.0
    yaw = 0.0
    pts = []
    for tid in token_ids:
        if tid < 0 or tid >= codebook.shape[0]:
            raise ValueError(f"token id out of range: {tid}")
        chunk = codebook[tid]  # (5,3)
        for k in range(5):
            dx, dy, dphi = float(chunk[k, 0]), float(chunk[k, 1]), float(chunk[k, 2])
            if rotate_by_yaw:
                # rotate (dx,dy) by current yaw before accumulating
                c, s = math.cos(yaw), math.sin(yaw)
                dxr = c * dx - s * dy
                dyr = s * dx + c * dy
                xs += dxr
                ys += dyr
                yaw += dphi
            else:
                xs += dx
                ys += dy
                yaw += dphi
            pts.append([xs, ys])
    return np.asarray(pts, dtype=np.float32)


def align_past_to_origin(past_xy: np.ndarray) -> np.ndarray:
    """
    shift past so that the last point becomes (0,0)
    """
    if past_xy.size == 0:
        return past_xy
    end = past_xy[-1].copy()
    return past_xy - end


# -------------------------
# Build the exact user_text prompt like training
# -------------------------
def build_user_text(ex: Dict[str, Any], past_tokens: List[int], prefix: str, past_len: int) -> str:
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

    past_str = tokens_to_traj_str(past_tokens, prefix=prefix)

    user_text = (
        "You are an autonomous driving agent.\n"
        "You are given THREE camera images: middle, left, right.\n"
        f"Driving prompt (intent): {prompt}\n"
        f"Vehicle state: {state_str}\n"
        f"Past trajectory tokens (past 3.0s, dt=0.5s, {past_len} tokens): {past_str}\n\n"
        "Task: Predict the future trajectory tokens for the next 5.0s (dt=0.5s, 10 tokens).\n"
        "Output ONLY the 10 tokens, separated by spaces. Do not output any other words.\n"
    )
    return user_text


# -------------------------
# Inference (single sample)
# -------------------------
@torch.inference_mode()
def infer_one(
    model,
    processor,
    tokenizer,
    img_mid: Image.Image,
    img_left: Image.Image,
    img_right: Image.Image,
    user_text: str,
    prefix: str,
    future_len: int,
    max_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Tuple[str, List[int]]:
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

    inputs = processor(
        text=[prompt_text],
        images=[[img_mid, img_left, img_right]],  # order must match training: mid, left, right
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    # move tensors to model device (device_map="auto" -> inputs can stay on CPU; transformers will dispatch,
    # but moving input_ids to first device is usually safe)
    if torch.cuda.is_available():
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=1,
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))

    out_ids = model.generate(**inputs, **gen_kwargs)
    out_text = tokenizer.decode(out_ids[0], skip_special_tokens=False)

    toks = parse_traj_tokens_from_text(out_text, prefix=prefix)
    toks = toks[:future_len]  # take first 10 tokens

    return out_text, toks


# -------------------------
# Plotting
# -------------------------
def plot_xy(
    past_xy: np.ndarray,
    gt_future_xy: np.ndarray,
    pred_future_xy: np.ndarray,
    save_path: str,
    title: str = "",
):
    plt.figure(figsize=(7, 7))
    # history
    if past_xy is not None and past_xy.size > 0:
        plt.plot(past_xy[:, 0], past_xy[:, 1], label="history (past)", linewidth=2)
    # gt
    if gt_future_xy is not None and gt_future_xy.size > 0:
        plt.plot(gt_future_xy[:, 0], gt_future_xy[:, 1], label="GT future", linewidth=2)
    # pred
    if pred_future_xy is not None and pred_future_xy.size > 0:
        plt.plot(pred_future_xy[:, 0], pred_future_xy[:, 1], label="Pred future", linewidth=2)

    # origin
    plt.scatter([0.0], [0.0], s=60, marker="x", label="t=0")

    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    if title:
        plt.title(title)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--cuda_visible_devices", type=str, default="0,1")
    ap.add_argument("--model_dir", required=True, help="ckpt dir for inference (e.g. /.../ckpt_grouped_full)")
    ap.add_argument("--processor_dir", default=None, help="If omitted, use model_dir.")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--base_dir", required=True)

    ap.add_argument("--codebook_npy", required=True, help="np.load -> (K,5,3) or (K,15)")
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--prefix", type=str, default="traj")

    ap.add_argument("--token_ctx_key", type=str, default="token_ctx_dt0p5_past3s_future5s")
    ap.add_argument("--img_left_key", type=str, default="front_left")
    ap.add_argument("--img_mid_key", type=str, default="zed_left")
    ap.add_argument("--img_right_key", type=str, default="front_right")

    ap.add_argument("--past_len", type=int, default=6)
    ap.add_argument("--future_len", type=int, default=10)

    ap.add_argument("--max_length", type=int, default=2048)

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--start_idx", type=int, default=0)

    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--rotate_by_yaw", action="store_true",
                    help="If codebook dx,dy are in body frame per step, enable this.")
    ap.add_argument("--enable_gc", action="store_true")

    args = ap.parse_args()

    # paths
    model_dir = str(Path(args.model_dir).expanduser().resolve())
    processor_dir = str(Path(args.processor_dir).expanduser().resolve()) if args.processor_dir else model_dir
    jsonl_path = str(Path(args.jsonl).expanduser().resolve())
    base_dir = str(Path(args.base_dir).expanduser().resolve())
    out_dir = str(Path(args.out_dir).expanduser().resolve())
    os.makedirs(out_dir, exist_ok=True)

    print(f"[CUDA_VISIBLE_DEVICES] {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    if torch.cuda.is_available():
        print(f"[CUDA] visible device count = {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - cuda:{i} = {torch.cuda.get_device_name(i)}")

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model_dir not found: {model_dir}")
    if not os.path.isdir(processor_dir):
        raise FileNotFoundError(f"processor_dir not found: {processor_dir}")
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"jsonl not found: {jsonl_path}")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"base_dir not found: {base_dir}")

    codebook = load_codebook(args.codebook_npy, K=args.K)
    print("[Codebook] shape =", codebook.shape, " (K,5,3)")

    # load processor/model
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
    model.config.use_cache = True  # inference keep cache

    # ensure vocab matches (important if you resized during training)
    if getattr(model.config, "vocab_size", None) != len(tokenizer):
        print(f"[Warn] vocab mismatch: model={model.config.vocab_size} tokenizer={len(tokenizer)} -> resizing")
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    index = JsonlOffsetIndex(jsonl_path)

    end_idx = min(len(index), args.start_idx + args.num_samples)
    print(f"[Infer] range: [{args.start_idx}, {end_idx}) total={len(index)}")

    for idx in range(args.start_idx, end_idx):
        ex = index.get(idx)

        # tokens
        ctx = ex.get(args.token_ctx_key, {})
        if not isinstance(ctx, dict):
            print(f"[Skip {idx}] missing token_ctx_key={args.token_ctx_key}")
            continue
        tp = ctx.get("tokens_past_3s", None)
        tf = ctx.get("tokens_future_5s", None)
        if not (isinstance(tp, list) and isinstance(tf, list)):
            print(f"[Skip {idx}] bad tokens fields")
            continue
        if len(tp) != args.past_len or len(tf) != args.future_len:
            print(f"[Skip {idx}] bad token length: past={len(tp)} future={len(tf)}")
            continue

        # images
        paths = ex.get("paths", {})
        if not isinstance(paths, dict):
            print(f"[Skip {idx}] paths not dict")
            continue

        p_left = _join_base(base_dir, paths.get(args.img_left_key))
        p_mid = _join_base(base_dir, paths.get(args.img_mid_key))
        p_right = _join_base(base_dir, paths.get(args.img_right_key))
        if any([(p is None) or (not os.path.exists(p)) for p in [p_left, p_mid, p_right]]):
            print(f"[Skip {idx}] missing image: {p_left}, {p_mid}, {p_right}")
            continue

        img_left = open_rgb(p_left)
        img_mid = open_rgb(p_mid)
        img_right = open_rgb(p_right)

        # build prompt text exactly like training
        user_text = build_user_text(ex, tp, prefix=args.prefix, past_len=args.past_len)

        # infer
        full_out_text, pred_tokens = infer_one(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            img_mid=img_mid,
            img_left=img_left,
            img_right=img_right,
            user_text=user_text,
            prefix=args.prefix,
            future_len=args.future_len,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # If model output fewer than future_len tokens, pad by repeating last or skip
        if len(pred_tokens) < args.future_len:
            if len(pred_tokens) == 0:
                print(f"[Warn {idx}] model produced 0 traj tokens. Skipping plot.")
                continue
            pred_tokens = pred_tokens + [pred_tokens[-1]] * (args.future_len - len(pred_tokens))

        # decode past/future to 10Hz XY
        past_xy = integrate_token_sequence_to_xy_10hz(
            tp, codebook, rotate_by_yaw=args.rotate_by_yaw
        )
        past_xy = align_past_to_origin(past_xy)

        gt_future_xy = integrate_token_sequence_to_xy_10hz(
            tf, codebook, rotate_by_yaw=args.rotate_by_yaw
        )

        pred_future_xy = integrate_token_sequence_to_xy_10hz(
            pred_tokens, codebook, rotate_by_yaw=args.rotate_by_yaw
        )

        # save plot
        prompt = ex.get("prompt", "")
        title = f"idx={idx} | prompt={prompt}"
        save_path = os.path.join(out_dir, f"traj_xy_idx{idx:06d}.png")
        plot_xy(past_xy, gt_future_xy, pred_future_xy, save_path, title=title)

        # also save tokens for debugging
        tok_path = os.path.join(out_dir, f"traj_tokens_idx{idx:06d}.json")
        with open(tok_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "idx": idx,
                    "prompt": prompt,
                    "past_tokens": tp,
                    "gt_future_tokens": tf,
                    "pred_future_tokens": pred_tokens,
                    "raw_model_text": full_out_text[-4000:],  # tail for debug
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"[OK] idx={idx} saved: {save_path}")

        if args.enable_gc:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("✅ done.")


if __name__ == "__main__":
    main()

