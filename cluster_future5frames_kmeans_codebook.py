#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

EARTH_R = 6378137.0  # meters


# -----------------------------
# helpers
# -----------------------------
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def is_all_zero(v: np.ndarray, eps: float = 1e-6) -> bool:
    return bool(np.max(np.abs(v)) <= eps)


def rot2(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def get_stamp_s(rec: Dict[str, Any]) -> Optional[float]:
    """best-effort timestamp extraction (seconds)."""
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
    """
    imu.810.AngleHeading (degrees).
    Assume 0°=North, 90°=East.
    """
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
    """small displacement between lat/lon -> (dE, dN) meters"""
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
    world EN -> ego@current
    Ego: +x forward, +y left
    forward in EN = [sin(yaw), cos(yaw)]
    left in EN    = [cos(yaw), -sin(yaw)]
    """
    fx, fn = math.sin(yaw0), math.cos(yaw0)
    lx, ln = math.cos(yaw0), -math.sin(yaw0)
    dx = dE * fx + dN * fn
    dy = dE * lx + dN * ln
    return dx, dy


# -----------------------------
# token building
# -----------------------------
def build_future5step_token(
    steps_EN: np.ndarray,   # [n-1,2]   step k = k->k+1 in EN
    yaws: np.ndarray,       # [n]       yaw in rad
    stamps: np.ndarray,     # [n]
    i: int,
    max_gap_s: float,
    max_step_m: float,
    max_yaw_rate: float,
) -> Optional[np.ndarray]:
    """
    For a given index i, build 5-step token using steps i..i+4 (needs i+5 exists).
    dx,dy are expressed in ego@i (yaw_i).
    dphi uses per-step wrap(yaw[k+1]-yaw[k]).
    Time constraint: each adjacent dt <= max_gap_s in these 5 steps.
    Abnormal constraints: step distance <= max_step_m, yaw rate <= max_yaw_rate.
    Return 15D vector or None if invalid.
    """
    n = len(yaws)
    if i + 5 >= n:
        return None

    # check adjacency in window edges i..i+4 (5 steps)
    for k in range(i, i + 5):
        dt = float(stamps[k + 1] - stamps[k])
        if (not np.isfinite(dt)) or dt <= 0 or dt > max_gap_s:
            return None

        dE, dN = float(steps_EN[k, 0]), float(steps_EN[k, 1])
        dist = math.hypot(dE, dN)
        if (not np.isfinite(dist)) or dist > max_step_m:
            return None

        dyaw = wrap_pi(float(yaws[k + 1] - yaws[k]))
        rate = abs(dyaw) / dt
        if (not np.isfinite(rate)) or rate > max_yaw_rate:
            return None

    yaw_i = float(yaws[i])
    if not np.isfinite(yaw_i):
        return None

    v = np.zeros((5, 3), dtype=np.float32)
    for t in range(5):
        k = i + t
        dE, dN = float(steps_EN[k, 0]), float(steps_EN[k, 1])
        dx, dy = EN_to_ego_xy(dE, dN, yaw_i)  # ego@i
        dphi = wrap_pi(float(yaws[k + 1] - yaws[k]))
        v[t, 0] = float(dx)
        v[t, 1] = float(dy)
        v[t, 2] = float(dphi)

    return v.reshape(-1)  # 15D


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument("--output_codebook", type=str, required=True)
    ap.add_argument("--output_summary", type=str, default="kmeans_summary_future5frames_dt0p5.json")

    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_train_tokens", type=int, default=120000)

    # time/abnormal constraints
    ap.add_argument("--max_gap_s", type=float, default=0.3,
                    help="Max adjacent dt allowed inside the 5-step token window. 10Hz nominal=0.1s; default 0.2s.")
    ap.add_argument("--max_step_m", type=float, default=3.0,
                    help="Max displacement per 0.1s step (meters). default 3.0m (30m/s) quite loose.")
    ap.add_argument("--max_yaw_rate", type=float, default=3.0,
                    help="Max yaw rate rad/s per step. default 3 rad/s (~172 deg/s) loose.")

    ap.add_argument("--zero_eps", type=float, default=1e-6)

    # Optional: emphasize dphi in clustering
    ap.add_argument("--phi_scale", type=float, default=1.0,
                    help="Multiply all dphi dims before standardization+clustering.")

    # MiniBatchKMeans knobs
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--reassign_ratio", type=float, default=0.01)

    args = ap.parse_args()

    in_path = Path(args.input_jsonl).expanduser().resolve()
    out_path = Path(args.output_jsonl).expanduser().resolve()
    cb_path = Path(args.output_codebook).expanduser().resolve()
    sum_path = Path(args.output_summary).expanduser().resolve()

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cb_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    token_dim = 15
    dphi_idx = np.array([2, 5, 8, 11, 14], dtype=np.int64)

    # -----------------------------
    # read + group by seq
    # -----------------------------
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

    # -----------------------------
    # Pass 1: reservoir sample tokens for clustering
    # -----------------------------
    reservoir = np.zeros((args.max_train_tokens, token_dim), dtype=np.float32)
    res_count = 0
    seen_nonzero = 0

    total_tokens_built = 0
    total_tokens_kept_for_train = 0
    total_tokens_zero = 0
    total_tokens_dropped = 0

    def maybe_add(v: np.ndarray):
        nonlocal res_count, seen_nonzero
        seen_nonzero += 1
        if res_count < args.max_train_tokens:
            reservoir[res_count] = v
            res_count += 1
        else:
            j = int(rng.integers(0, seen_nonzero))
            if j < args.max_train_tokens:
                reservoir[j] = v

    # also cache per-record token vector so we can encode later without recompute (optional)
    # but for memory safety, we recompute in pass2.

    for seq, recs in tqdm(seq_map.items(), desc="Pass1 collect"):
        # extract stamps/pose/yaw, sort by time
        stamps = np.array([get_stamp_s(r) for r in recs], dtype=np.float64)
        ok_stamp = np.isfinite(stamps)
        if not bool(np.all(ok_stamp)):
            continue

        order = np.argsort(stamps)
        recs = [recs[i] for i in order]
        stamps = stamps[order]

        latlons = [get_lat_lon(r) for r in recs]
        yaws = np.array([get_heading_rad(r) if get_heading_rad(r) is not None else np.nan for r in recs], dtype=np.float64)

        n = len(recs)
        if n < 6:
            continue

        valid_pose = np.array([(latlons[i] is not None) and np.isfinite(yaws[i]) for i in range(n)], dtype=bool)
        if not bool(np.all(valid_pose)):
            # still can proceed: we will drop indices that touch invalid
            pass

        # steps EN
        steps_EN = np.zeros((n - 1, 2), dtype=np.float64)
        for k in range(n - 1):
            if not (valid_pose[k] and valid_pose[k + 1]):
                steps_EN[k] = np.nan
                continue
            lat0, lon0 = latlons[k]      # type: ignore
            lat1, lon1 = latlons[k + 1]  # type: ignore
            dE, dN = latlon_to_EN_step(lat0, lon0, lat1, lon1)
            steps_EN[k, 0] = dE
            steps_EN[k, 1] = dN

        for i in range(n):
            v = build_future5step_token(
                steps_EN=steps_EN,
                yaws=yaws,
                stamps=stamps,
                i=i,
                max_gap_s=args.max_gap_s,
                max_step_m=args.max_step_m,
                max_yaw_rate=args.max_yaw_rate,
            )
            if v is None:
                total_tokens_dropped += 1
                continue

            total_tokens_built += 1

            if is_all_zero(v, eps=args.zero_eps):
                total_tokens_zero += 1
                continue

            # phi scale in token space before clustering
            v2 = v.astype(np.float32).copy()
            if args.phi_scale != 1.0:
                v2[dphi_idx] *= float(args.phi_scale)

            maybe_add(v2)
            total_tokens_kept_for_train += 1

    if res_count == 0:
        raise RuntimeError("No non-zero tokens collected for clustering. Check filters or data.")

    X_train = reservoir[:res_count].astype(np.float32)
    print("\n[Pass1 Stats]")
    print(f"  total_lines={total_lines}")
    print(f"  total_tokens_built={total_tokens_built}")
    print(f"  total_tokens_zero={total_tokens_zero}")
    print(f"  total_tokens_dropped={total_tokens_dropped}")
    print(f"  train_sampled_nonzero={res_count} (reservoir), seen_nonzero={seen_nonzero}")

    # -----------------------------
    # KMeans build codebook:
    # code 0 = all-zero
    # codes 1..K-1 = kmeans clusters on nonzero tokens
    # -----------------------------
    k_total = int(args.k)
    if k_total < 2:
        raise ValueError("k must be >= 2")
    k_nonzero = k_total - 1

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X_train.astype(np.float64)).astype(np.float32)

    kmeans = MiniBatchKMeans(
        n_clusters=k_nonzero,
        random_state=args.seed,
        batch_size=int(args.batch_size),
        max_iter=int(args.max_iter),
        n_init=int(args.n_init),
        reassignment_ratio=float(args.reassign_ratio),
        verbose=0,
    )

    print(f"\n[KMeans] fitting MiniBatchKMeans: K_nonzero={k_nonzero}, token_dim={token_dim}")
    kmeans.fit(Xs)
    inertia = float(kmeans.inertia_)
    print(f"[KMeans] done. inertia={inertia:.6f}")

    centers_scaled = kmeans.cluster_centers_.astype(np.float32)  # in standardized space
    centers_phiScaled = scaler.inverse_transform(centers_scaled).astype(np.float32)  # in phi_scaled token space

    centers_phiScaled_full = np.zeros((k_total, token_dim), dtype=np.float32)
    centers_phiScaled_full[0] = 0.0
    centers_phiScaled_full[1:] = centers_phiScaled

    centers_unscaled = centers_phiScaled_full.copy()
    if args.phi_scale != 1.0:
        centers_unscaled[:, dphi_idx] /= float(args.phi_scale)

    # -----------------------------
    # Pass 2: encode each record (if token valid)
    # write output jsonl with token_id + raw 15D vec (unscaled)
    # -----------------------------
    def encode_token(v_unscaled: np.ndarray) -> int:
        if is_all_zero(v_unscaled, eps=args.zero_eps):
            return 0
        v = v_unscaled.astype(np.float32).copy()
        if args.phi_scale != 1.0:
            v[dphi_idx] *= float(args.phi_scale)
        vs = scaler.transform(v[None, :].astype(np.float64)).astype(np.float32)
        cid = int(kmeans.predict(vs)[0])  # 0..k_nonzero-1
        return cid + 1

    kept_records = 0
    dropped_records = 0

    with out_path.open("w", encoding="utf-8") as fo:
        for seq, recs in tqdm(seq_map.items(), desc="Pass2 write"):
            stamps = np.array([get_stamp_s(r) for r in recs], dtype=np.float64)
            if not bool(np.all(np.isfinite(stamps))):
                continue
            order = np.argsort(stamps)
            recs = [recs[i] for i in order]
            stamps = stamps[order]

            latlons = [get_lat_lon(r) for r in recs]
            yaws = np.array([get_heading_rad(r) if get_heading_rad(r) is not None else np.nan for r in recs], dtype=np.float64)
            n = len(recs)
            if n < 6:
                continue

            valid_pose = np.array([(latlons[i] is not None) and np.isfinite(yaws[i]) for i in range(n)], dtype=bool)
            steps_EN = np.zeros((n - 1, 2), dtype=np.float64)
            for k in range(n - 1):
                if not (valid_pose[k] and valid_pose[k + 1]):
                    steps_EN[k] = np.nan
                    continue
                lat0, lon0 = latlons[k]      # type: ignore
                lat1, lon1 = latlons[k + 1]  # type: ignore
                dE, dN = latlon_to_EN_step(lat0, lon0, lat1, lon1)
                steps_EN[k, 0] = dE
                steps_EN[k, 1] = dN

            for i in range(n):
                v = build_future5step_token(
                    steps_EN=steps_EN,
                    yaws=yaws,
                    stamps=stamps,
                    i=i,
                    max_gap_s=args.max_gap_s,
                    max_step_m=args.max_step_m,
                    max_yaw_rate=args.max_yaw_rate,
                )
                if v is None:
                    dropped_records += 1
                    continue

                token_id = encode_token(v)

                rec = recs[i]
                rec["action_token_dt0p5"] = {
                    "dt_token": 0.5,
                    "hz_src": 10.0,
                    "steps": 5,
                    "token_dim": 15,
                    "frame": "ego_at_current_time",
                    "desc": "5 successive step deltas (dx,dy,dphi) for next 0.5s, all in ego@current.",
                    "vec_unscaled": v.astype(float).tolist(),
                    "token_id": int(token_id),
                    "codebook": cb_path.name,
                    "method": "MiniBatchKMeans+StandardScaler",
                    "phi_scale_used_in_training": float(args.phi_scale),
                }

                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept_records += 1

    # -----------------------------
    # save codebook
    # -----------------------------
    dims = []
    for _ in range(5):
        dims += ["dx_m", "dy_m", "dphi_rad"]

    codebook = {
        "name": cb_path.name,
        "method": "MiniBatchKMeans",
        "k": int(k_total),
        "dt_token": 0.5,
        "hz_src": 10.0,
        "chunk_steps": 5,
        "token_dim": 15,
        "dims": dims,
        "note": "code 0 reserved as all-zero token; codes 1..K-1 are kmeans clusters on nonzero tokens.",
        "phi_scale": float(args.phi_scale),

        "filters": {
            "max_gap_s": float(args.max_gap_s),
            "max_step_m": float(args.max_step_m),
            "max_yaw_rate": float(args.max_yaw_rate),
            "zero_eps": float(args.zero_eps),
        },

        "scaler": {
            "type": "StandardScaler",
            "with_mean": True,
            "with_std": True,
            "mean": scaler.mean_.astype(float).tolist(),
            "scale": scaler.scale_.astype(float).tolist(),
        },

        "kmeans": {
            "type": "MiniBatchKMeans",
            "n_clusters_nonzero": int(k_nonzero),
            "batch_size": int(args.batch_size),
            "max_iter": int(args.max_iter),
            "n_init": int(args.n_init),
            "reassignment_ratio": float(args.reassign_ratio),
            "random_state": int(args.seed),
            "inertia": float(inertia),
        },

        # centers for reconstruction (true units dx/dy/dphi)
        "centers_unscaled_for_reconstruction": centers_unscaled.astype(float).tolist(),
        # centers in phi-scaled token space (debug)
        "centers_phiScaled_token_space": centers_phiScaled_full.astype(float).tolist(),

        "train_sampled_nonzero_tokens": int(res_count),
        "train_seen_nonzero_tokens": int(seen_nonzero),
        "seed": int(args.seed),
    }

    with cb_path.open("w", encoding="utf-8") as f:
        json.dump(codebook, f, ensure_ascii=False, indent=2)

    summary = {
        "input": str(in_path),
        "output_jsonl": str(out_path),
        "output_codebook": str(cb_path),
        "total_lines": int(total_lines),
        "pass1_tokens_built": int(total_tokens_built),
        "pass1_tokens_zero": int(total_tokens_zero),
        "pass1_tokens_dropped": int(total_tokens_dropped),
        "train_sampled": int(res_count),
        "kept_records_with_token": int(kept_records),
        "dropped_records_no_token": int(dropped_records),
        "k": int(k_total),
        "kmeans_inertia": float(inertia),
        "phi_scale": float(args.phi_scale),
    }

    with sum_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Output jsonl:    {out_path}")
    print(f"Output codebook: {cb_path}")
    print(f"Summary:         {sum_path}")
    print(f"Kept={kept_records} Dropped={dropped_records}")


if __name__ == "__main__":
    main()
