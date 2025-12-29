#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

EARTH_R = 6378137.0  # meters


# -------------------------
# math utils
# -------------------------
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# -------------------------
# parsing utils
# -------------------------
def get_stamp_s(rec: Dict[str, Any]) -> Optional[float]:
    tr = rec.get("time_ros")
    if isinstance(tr, dict) and "secs" in tr and "nsecs" in tr:
        try:
            return float(tr["secs"]) + float(tr["nsecs"]) * 1e-9
        except Exception:
            pass
    canfd = rec.get("canfd")
    if isinstance(canfd, dict) and "stamp" in canfd:
        try:
            return float(canfd["stamp"])
        except Exception:
            pass
    imu = rec.get("imu")
    if isinstance(imu, dict) and "stamp" in imu:
        try:
            return float(imu["stamp"])
        except Exception:
            pass
    return None


def get_lat_lon(rec: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    imu = rec.get("imu")
    if not isinstance(imu, dict):
        return None
    nav804 = imu.get("804")
    if not isinstance(nav804, dict):
        return None
    if "PosLat" not in nav804 or "PosLon" not in nav804:
        return None
    try:
        return float(nav804["PosLat"]), float(nav804["PosLon"])
    except Exception:
        return None


def get_heading_rad(rec: Dict[str, Any]) -> Optional[float]:
    imu = rec.get("imu")
    if not isinstance(imu, dict):
        return None
    nav810 = imu.get("810")
    if not isinstance(nav810, dict):
        return None
    if "AngleHeading" not in nav810:
        return None
    try:
        hdg_deg = float(nav810["AngleHeading"]) % 360.0
        return math.radians(hdg_deg)
    except Exception:
        return None


def latlon_to_EN_step(lat0: float, lon0: float, lat1: float, lon1: float) -> Tuple[float, float]:
    lat0r = math.radians(lat0)
    lat1r = math.radians(lat1)
    dlat = lat1r - lat0r
    dlon = math.radians(lon1 - lon0)
    mean_lat = 0.5 * (lat0r + lat1r)
    dN = dlat * EARTH_R
    dE = dlon * EARTH_R * math.cos(mean_lat)
    return dE, dN


def EN_to_ego_xy(dE: float, dN: float, yaw0: float) -> Tuple[float, float]:
    """
    world EN -> ego@current yaw0
    Ego: +x forward, +y left
    forward in EN = [sin(yaw), cos(yaw)]
    left in EN    = [cos(yaw), -sin(yaw)]
    """
    fx, fn = math.sin(yaw0), math.cos(yaw0)
    lx, ln = math.cos(yaw0), -math.sin(yaw0)
    dx = dE * fx + dN * fn
    dy = dE * lx + dN * ln
    return dx, dy


# -------------------------
# codebook: nearest center (KMeans codebook with scaler)
# -------------------------
class CodebookKMeans:
    def __init__(self, cb: Dict[str, Any]):
        if "centers_unscaled_for_reconstruction" not in cb:
            raise KeyError("codebook missing centers_unscaled_for_reconstruction")
        self.centers = np.asarray(cb["centers_unscaled_for_reconstruction"], dtype=np.float32)  # [K,15]
        self.k = int(cb.get("k", self.centers.shape[0]))
        self.steps = int(cb.get("chunk_steps", 5))
        self.token_dim = int(cb.get("token_dim", 15))
        if self.centers.shape[1] != self.token_dim:
            raise ValueError("codebook centers dim mismatch")

        sc = cb.get("scaler", {})
        self.mean = np.asarray(sc.get("mean", np.zeros((self.token_dim,), dtype=np.float32)), dtype=np.float32)
        self.scale = np.asarray(sc.get("scale", np.ones((self.token_dim,), dtype=np.float32)), dtype=np.float32)

        self.phi_scale = float(cb.get("phi_scale", 1.0))
        self.dphi_idx = np.array([2, 5, 8, 11, 14], dtype=np.int64)

        centers_phiScaled = self.centers.copy()
        if self.phi_scale != 1.0:
            centers_phiScaled[:, self.dphi_idx] *= self.phi_scale

        self.centers_std = (centers_phiScaled - self.mean[None, :]) / (self.scale[None, :] + 1e-12)

    def encode(self, v15_unscaled: np.ndarray, zero_eps: float = 1e-6) -> int:
        if float(np.max(np.abs(v15_unscaled))) <= zero_eps:
            return 0

        v = v15_unscaled.astype(np.float32).copy()
        if self.phi_scale != 1.0:
            v[self.dphi_idx] *= self.phi_scale
        v_std = (v - self.mean) / (self.scale + 1e-12)

        diff = self.centers_std - v_std[None, :]
        d2 = np.sum(diff * diff, axis=1)
        return int(np.argmin(d2))


# -------------------------
# build 15D token vector (5 steps) in ego@anchor_i
# -------------------------
def build_token_vec_5steps_in_ego_anchor(
    steps_EN: np.ndarray,   # [n-1,2]
    yaws: np.ndarray,       # [n]
    stamps: np.ndarray,     # [n]
    anchor_i: int,          # ego frame anchor
    start_k: int,           # step start index in world sequence (k->k+1)
    max_gap_s: float,
    max_step_m: float,
    max_yaw_rate: float,
) -> Optional[np.ndarray]:
    """
    Build vec15 = 5*(dx,dy,dphi) for steps start_k..start_k+4,
    with dx,dy expressed in ego@anchor_i (NOT ego@start_k).
    dphi is yaw[k+1]-yaw[k] (world).
    """
    n = len(yaws)
    if start_k + 5 >= n:
        return None

    yaw_anchor = float(yaws[anchor_i])
    if not np.isfinite(yaw_anchor):
        return None

    v = np.zeros((5, 3), dtype=np.float32)

    for t in range(5):
        k = start_k + t

        dt = float(stamps[k + 1] - stamps[k])
        if (not np.isfinite(dt)) or dt <= 0 or dt > max_gap_s:
            return None

        dE, dN = float(steps_EN[k, 0]), float(steps_EN[k, 1])
        if not (np.isfinite(dE) and np.isfinite(dN)):
            return None
        dist = math.hypot(dE, dN)
        if (not np.isfinite(dist)) or dist > max_step_m:
            return None

        dyaw = wrap_pi(float(yaws[k + 1] - yaws[k]))
        rate = abs(dyaw) / dt
        if (not np.isfinite(rate)) or rate > max_yaw_rate:
            return None

        dx, dy = EN_to_ego_xy(dE, dN, yaw_anchor)
        v[t, 0] = float(dx)
        v[t, 1] = float(dy)
        v[t, 2] = float(dyaw)

    return v.reshape(-1)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--codebook", type=str, required=True)
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument("--output_summary", type=str, default="ctx_tokens_only_summary.json")

    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--token_dt", type=float, default=0.5)  # 5 frames
    ap.add_argument("--future_sec", type=float, default=5.0)
    ap.add_argument("--past_sec", type=float, default=3.0)

    ap.add_argument("--max_gap_s", type=float, default=0.5)
    ap.add_argument("--max_step_m", type=float, default=3.0)
    ap.add_argument("--max_yaw_rate", type=float, default=3.0)
    ap.add_argument("--zero_eps", type=float, default=1e-6)

    ap.add_argument("--keep_vecs", action="store_true",
                    help="also save raw 15D vectors for debugging (bigger output).")

    args = ap.parse_args()

    in_path = Path(args.input_jsonl).expanduser().resolve()
    cb_path = Path(args.codebook).expanduser().resolve()
    out_path = Path(args.output_jsonl).expanduser().resolve()
    sum_path = Path(args.output_summary).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(in_path)
    if not cb_path.exists():
        raise FileNotFoundError(cb_path)

    steps_per_token = int(round(args.token_dt * args.hz))  # 0.5*10=5
    if steps_per_token != 5:
        raise ValueError("This script assumes token_dt=0.5 and hz=10 => 5 steps per token.")

    fut_tokens = int(round(args.future_sec / args.token_dt))  # 10
    past_tokens = int(round(args.past_sec / args.token_dt))   # 6
    fut_steps = fut_tokens * steps_per_token  # 50
    past_steps = past_tokens * steps_per_token  # 30

    with cb_path.open("r", encoding="utf-8") as f:
        cbj = json.load(f)
    cb = CodebookKMeans(cbj)

    # read & group by seq
    seq_map: Dict[str, List[Dict[str, Any]]] = {}
    total_lines = 0
    with in_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Read jsonl"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seq = rec.get("_seq", "UNKNOWN_SEQ")
            seq_map.setdefault(seq, []).append(rec)
            total_lines += 1

    kept = 0
    dropped = 0
    reasons = {
        "missing_stamp": 0,
        "missing_pose_or_yaw": 0,
        "window_oob": 0,
        "gap_or_abnormal": 0,
        "token_build_fail": 0,
    }

    with out_path.open("w", encoding="utf-8") as fo:
        for seq, recs in tqdm(seq_map.items(), desc="Process seq"):
            stamps = np.array([get_stamp_s(r) for r in recs], dtype=np.float64)
            if not bool(np.all(np.isfinite(stamps))):
                dropped += len(recs)
                reasons["missing_stamp"] += len(recs)
                continue

            order = np.argsort(stamps)
            recs = [recs[i] for i in order]
            stamps = stamps[order]
            n = len(recs)

            latlons = [get_lat_lon(r) for r in recs]
            yaws = np.array([get_heading_rad(r) if get_heading_rad(r) is not None else np.nan for r in recs], dtype=np.float64)
            valid_pose = np.array([(latlons[i] is not None) and np.isfinite(yaws[i]) for i in range(n)], dtype=bool)

            if n < (past_steps + fut_steps + 1):
                continue

            # EN steps
            steps_EN = np.zeros((n - 1, 2), dtype=np.float64)
            steps_EN[:] = np.nan
            for k in range(n - 1):
                if not (valid_pose[k] and valid_pose[k + 1]):
                    continue
                lat0, lon0 = latlons[k]      # type: ignore
                lat1, lon1 = latlons[k + 1]  # type: ignore
                dE, dN = latlon_to_EN_step(lat0, lon0, lat1, lon1)
                steps_EN[k, 0] = dE
                steps_EN[k, 1] = dN

            # per-frame
            for i in range(n):
                if not valid_pose[i]:
                    dropped += 1
                    reasons["missing_pose_or_yaw"] += 1
                    continue

                start = i - past_steps
                end = i + fut_steps
                if start < 0 or end >= n:
                    dropped += 1
                    reasons["window_oob"] += 1
                    continue

                # check window edges: dt, step, yaw rate, finite
                ok = True
                for k in range(start, end):
                    dt = float(stamps[k + 1] - stamps[k])
                    if (not np.isfinite(dt)) or dt <= 0 or dt > args.max_gap_s:
                        ok = False
                        break
                    if not (np.isfinite(steps_EN[k, 0]) and np.isfinite(steps_EN[k, 1])):
                        ok = False
                        break
                    dist = math.hypot(float(steps_EN[k, 0]), float(steps_EN[k, 1]))
                    if (not np.isfinite(dist)) or dist > args.max_step_m:
                        ok = False
                        break
                    dyaw = wrap_pi(float(yaws[k + 1] - yaws[k]))
                    rate = abs(dyaw) / dt
                    if (not np.isfinite(rate)) or rate > args.max_yaw_rate:
                        ok = False
                        break
                if not ok:
                    dropped += 1
                    reasons["gap_or_abnormal"] += 1
                    continue

                # future tokens: start_k = i + 0, i+5, ... i+45; anchor_i = i
                fut_ids: List[int] = []
                fut_vecs: List[List[float]] = []
                for t in range(fut_tokens):
                    start_k = i + t * steps_per_token
                    v15 = build_token_vec_5steps_in_ego_anchor(
                        steps_EN=steps_EN, yaws=yaws, stamps=stamps,
                        anchor_i=i, start_k=start_k,
                        max_gap_s=args.max_gap_s, max_step_m=args.max_step_m, max_yaw_rate=args.max_yaw_rate
                    )
                    if v15 is None:
                        ok = False
                        break
                    tid = cb.encode(v15, zero_eps=args.zero_eps)
                    fut_ids.append(tid)
                    if args.keep_vecs:
                        fut_vecs.append(v15.astype(float).tolist())

                if not ok or len(fut_ids) != fut_tokens:
                    dropped += 1
                    reasons["token_build_fail"] += 1
                    continue

                # past tokens: represent past in ego@i too.
                # We take 6 chunks covering [i-30..i], with start_k = i-30, i-25, ..., i-5; anchor_i = i
                past_ids: List[int] = []
                past_vecs: List[List[float]] = []
                for t in range(past_tokens):
                    start_k = (i - past_steps) + t * steps_per_token  # i-30, i-25, ..., i-5
                    v15 = build_token_vec_5steps_in_ego_anchor(
                        steps_EN=steps_EN, yaws=yaws, stamps=stamps,
                        anchor_i=i, start_k=start_k,
                        max_gap_s=args.max_gap_s, max_step_m=args.max_step_m, max_yaw_rate=args.max_yaw_rate
                    )
                    if v15 is None:
                        ok = False
                        break
                    tid = cb.encode(v15, zero_eps=args.zero_eps)
                    past_ids.append(tid)
                    if args.keep_vecs:
                        past_vecs.append(v15.astype(float).tolist())

                if not ok or len(past_ids) != past_tokens:
                    dropped += 1
                    reasons["token_build_fail"] += 1
                    continue

                rec = recs[i]
                rec["token_ctx_dt0p5_past3s_future5s"] = {
                    "hz": float(args.hz),
                    "token_dt": float(args.token_dt),
                    "steps_per_token": int(steps_per_token),
                    "past_sec": float(args.past_sec),
                    "future_sec": float(args.future_sec),
                    "past_tokens": int(past_tokens),
                    "future_tokens": int(fut_tokens),
                    "coord_frame": "ego_at_current_time(+x forward,+y left)",
                    "codebook": cb_path.name,
                    "tokens_past_3s": past_ids,      # 6
                    "tokens_future_5s": fut_ids,     # 10
                    "filters": {
                        "max_gap_s": float(args.max_gap_s),
                        "max_step_m": float(args.max_step_m),
                        "max_yaw_rate": float(args.max_yaw_rate),
                        "zero_eps": float(args.zero_eps),
                    },
                }
                if args.keep_vecs:
                    rec["token_ctx_dt0p5_past3s_future5s"]["past_vec15_unscaled"] = past_vecs
                    rec["token_ctx_dt0p5_past3s_future5s"]["future_vec15_unscaled"] = fut_vecs

                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

    summary = {
        "input": str(in_path),
        "codebook": str(cb_path),
        "output": str(out_path),
        "total_lines": int(total_lines),
        "kept": int(kept),
        "dropped": int(dropped),
        "drop_reasons": reasons,
        "config": {
            "hz": float(args.hz),
            "token_dt": float(args.token_dt),
            "past_sec": float(args.past_sec),
            "future_sec": float(args.future_sec),
            "past_tokens": int(past_tokens),
            "future_tokens": int(fut_tokens),
            "max_gap_s": float(args.max_gap_s),
            "max_step_m": float(args.max_step_m),
            "max_yaw_rate": float(args.max_yaw_rate),
            "zero_eps": float(args.zero_eps),
            "keep_vecs": bool(args.keep_vecs),
        },
    }
    with sum_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Output jsonl: {out_path}")
    print(f"Summary json: {sum_path}")
    print(f"Kept={kept} Dropped={dropped}")
    print("Drop reasons:", reasons)


if __name__ == "__main__":
    main()
