#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
from tqdm import tqdm

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None


GEN4_CLASS_NAMES = {
    0: "pedestrian",
    1: "two-wheeler",
    2: "car",
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    preprocess_family: str
    sensor_w: int
    sensor_h: int
    event_suffix: str
    min_diag: float
    min_side: float
    floor_train_box_width: bool
    keep_class_ids: Optional[Tuple[int, ...]]
    class_id_remap: Optional[Dict[int, int]]
    class_names: Optional[Dict[int, str]]
    include_box_timestamp: bool


DATASET_CONFIGS = {
    "gen1": DatasetConfig(
        name="gen1",
        preprocess_family="gen1",
        sensor_w=304,
        sensor_h=240,
        event_suffix="_td.dat.h5",
        min_diag=30.0,
        min_side=10.0,
        floor_train_box_width=True,
        keep_class_ids=None,
        class_id_remap=None,
        class_names=None,
        include_box_timestamp=False,
    ),
    "gen4": DatasetConfig(
        name="gen4",
        preprocess_family="gen4",
        sensor_w=1280,
        sensor_h=720,
        event_suffix="_td.h5",
        min_diag=5.0,
        min_side=5.0,
        floor_train_box_width=False,
        keep_class_ids=(0, 1, 2),
        class_id_remap={0: 0, 1: 1, 2: 2},
        class_names=GEN4_CLASS_NAMES,
        include_box_timestamp=True,
    ),
    "etram": DatasetConfig(
        name="etram",
        preprocess_family="gen4",
        sensor_w=1280,
        sensor_h=720,
        event_suffix="_td.h5",
        min_diag=5.0,
        min_side=5.0,
        floor_train_box_width=False,
        keep_class_ids=(0, 1, 2),
        class_id_remap={0: 0, 1: 1, 2: 2},
        class_names=GEN4_CLASS_NAMES,
        include_box_timestamp=True,
    ),
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _find_first_key(mapping: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in mapping:
            return mapping[key]
    return None


def canonical_box_dtype() -> np.dtype:
    return np.dtype([
        ("t", np.int64),
        ("x", np.float32),
        ("y", np.float32),
        ("w", np.float32),
        ("h", np.float32),
        ("class_id", np.int32),
        ("track_id", np.int32),
    ])


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


# -----------------------------------------------------------------------------
# Event loading
# -----------------------------------------------------------------------------

def load_events_h5(h5_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        containers = [f]
        for group_name in ["events", "event", "td", "data"]:
            if group_name in f and isinstance(f[group_name], h5py.Group):
                containers.append(f[group_name])

        x = y = p = t = None
        for g in containers:
            keys = {k.lower(): k for k in g.keys()}
            x_key = _find_first_key(keys, ["x"])
            y_key = _find_first_key(keys, ["y"])
            p_key = _find_first_key(keys, ["p", "pol", "polarity"])
            t_key = _find_first_key(keys, ["t", "ts", "timestamp", "timestamps", "time"])
            if x_key and y_key and p_key and t_key:
                x = g[x_key][()]
                y = g[y_key][()]
                p = g[p_key][()]
                t = g[t_key][()]
                break

    if x is None:
        raise KeyError(f"Cannot find x/y/p/t arrays in {h5_path}")

    x = np.asarray(x, dtype=np.int32).reshape(-1)
    y = np.asarray(y, dtype=np.int32).reshape(-1)
    t = np.asarray(t, dtype=np.int64).reshape(-1)
    p = np.asarray(p).reshape(-1)

    if p.dtype == np.bool_:
        p = np.where(p, 1, -1)
    else:
        p = np.where(p.astype(np.int16) > 0, 1, -1)
    p = p.astype(np.int8, copy=False)

    if not (len(x) == len(y) == len(p) == len(t)):
        raise ValueError("x/y/p/t lengths are inconsistent.")

    if len(t) > 1 and np.any(t[1:] < t[:-1]):
        order = np.argsort(t, kind="stable")
        x, y, p, t = x[order], y[order], p[order], t[order]

    return x, y, p, t


# -----------------------------------------------------------------------------
# Label loading
# -----------------------------------------------------------------------------

def canonicalize_boxes(arr: np.ndarray) -> np.ndarray:
    dtype = canonical_box_dtype()

    if arr.dtype.names is not None:
        fields = {name.lower(): name for name in arr.dtype.names}
        tf = _find_first_key(fields, ["t", "ts", "timestamp", "time"])
        xf = _find_first_key(fields, ["x", "left"])
        yf = _find_first_key(fields, ["y", "top"])
        wf = _find_first_key(fields, ["w", "width"])
        hf = _find_first_key(fields, ["h", "height"])
        cf = _find_first_key(fields, ["class_id", "class", "label"])
        trf = _find_first_key(fields, ["track_id", "track", "id"])

        required = [tf, xf, yf, wf, hf]
        if any(v is None for v in required):
            raise KeyError("bbox.npy dtype must contain time/x/y/w/h fields or aliases.")

        out = np.zeros((len(arr),), dtype=dtype)
        out["t"] = np.asarray(arr[tf], dtype=np.int64)
        out["x"] = np.asarray(arr[xf], dtype=np.float32)
        out["y"] = np.asarray(arr[yf], dtype=np.float32)
        out["w"] = np.asarray(arr[wf], dtype=np.float32)
        out["h"] = np.asarray(arr[hf], dtype=np.float32)
        out["class_id"] = np.asarray(arr[cf], dtype=np.int32) if cf is not None else -1
        out["track_id"] = np.asarray(arr[trf], dtype=np.int32) if trf is not None else -1
        return out

    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError("Non-structured bbox.npy must be shape [N, >=5].")

    out = np.zeros((arr.shape[0],), dtype=dtype)
    out["t"] = arr[:, 0].astype(np.int64)
    out["x"] = arr[:, 1].astype(np.float32)
    out["y"] = arr[:, 2].astype(np.float32)
    out["w"] = arr[:, 3].astype(np.float32)
    out["h"] = arr[:, 4].astype(np.float32)
    out["class_id"] = arr[:, 5].astype(np.int32) if arr.shape[1] > 5 else -1
    out["track_id"] = arr[:, 6].astype(np.int32) if arr.shape[1] > 6 else -1
    return out


def load_bbox_npy(bbox_path: str) -> np.ndarray:
    arr = np.load(bbox_path, allow_pickle=True)
    if arr is None or len(arr) == 0:
        return np.zeros((0,), dtype=canonical_box_dtype())
    return canonicalize_boxes(arr)


# -----------------------------------------------------------------------------
# Label filters
# -----------------------------------------------------------------------------

def crop_to_fov(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    if len(labels) == 0:
        return labels

    x0 = labels["x"].copy()
    y0 = labels["y"].copy()
    x1 = x0 + labels["w"]
    y1 = y0 + labels["h"]

    x0 = np.clip(x0, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    x1 = np.clip(x1, 0, width - 1)
    y1 = np.clip(y1, 0, height - 1)

    out = labels.copy()
    out["x"] = x0
    out["y"] = y0
    out["w"] = x1 - x0
    out["h"] = y1 - y0

    keep = (out["w"] > 0) & (out["h"] > 0)
    return out[keep]


def filter_min_boxes(labels: np.ndarray, min_diag: float, min_side: float) -> np.ndarray:
    if len(labels) == 0:
        return labels
    diag = np.sqrt(labels["w"] ** 2 + labels["h"] ** 2)
    keep = (diag >= min_diag) & (labels["w"] >= min_side) & (labels["h"] >= min_side)
    return labels[keep]


def filter_max_boxes_train(
    labels: np.ndarray,
    split: str,
    frame_width: int,
    floor_train_box_width: bool,
) -> np.ndarray:
    if len(labels) == 0 or split != "train":
        return labels
    if floor_train_box_width:
        max_width = (9 * int(frame_width)) // 10
    else:
        max_width = 0.9 * float(frame_width)
    keep = labels["w"] <= max_width
    return labels[keep]


def filter_class_ids(labels: np.ndarray, keep_ids: Optional[Sequence[int]]) -> np.ndarray:
    if len(labels) == 0 or keep_ids is None:
        return labels
    keep = np.isin(labels["class_id"], list(keep_ids))
    return labels[keep]


def remap_class_ids(labels: np.ndarray, remap: Optional[Dict[int, int]]) -> np.ndarray:
    if len(labels) == 0 or remap is None:
        return labels
    out = labels.copy()
    out["class_id"] = np.array([remap.get(int(v), -1) for v in labels["class_id"]], dtype=np.int32)
    keep = out["class_id"] >= 0
    return out[keep]


def mangle_labels(labels: np.ndarray, split: str, config: DatasetConfig, sensor_w: int, sensor_h: int) -> np.ndarray:
    labels = crop_to_fov(labels, width=sensor_w, height=sensor_h)
    labels = filter_min_boxes(labels, min_diag=config.min_diag, min_side=config.min_side)
    labels = filter_max_boxes_train(
        labels,
        split=split,
        frame_width=sensor_w,
        floor_train_box_width=config.floor_train_box_width,
    )
    labels = filter_class_ids(labels, keep_ids=config.keep_class_ids)
    labels = remap_class_ids(labels, remap=config.class_id_remap)
    return labels


def has_multiple_annotations_per_frame(frame_labels: np.ndarray) -> bool:
    if len(frame_labels) < 2:
        return False
    return len(np.unique(frame_labels["t"])) > 1


def cherry_pick_label_timestamps(frame_labels: np.ndarray) -> np.ndarray:
    times, counts = np.unique(frame_labels["t"], return_counts=True)
    max_count = np.max(counts)
    median_time = np.median(times)
    candidates = times[counts == max_count]
    picked_time = candidates[np.argmin(np.abs(candidates - median_time))]
    return frame_labels[frame_labels["t"] == picked_time]


# -----------------------------------------------------------------------------
# HTA-RGB representation core
# -----------------------------------------------------------------------------

def box_filter(img: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return img.astype(np.float32, copy=False)
    return cv2.blur(img.astype(np.float32, copy=False), (ksize, ksize), borderType=cv2.BORDER_REPLICATE)


def smooth_positive_saturate(x: np.ndarray, vmax: float) -> np.ndarray:
    vmax = max(float(vmax), 1e-6)
    return vmax * np.tanh(x / vmax)


def accumulate_counts(
    xs: np.ndarray,
    ys: np.ndarray,
    ps: np.ndarray,
    ts: np.ndarray,
    t0: int,
    t1: int,
    h: int,
    w: int,
    recent_gamma: float,
    recent_floor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.zeros((h, w), dtype=np.float32)
    neg = np.zeros((h, w), dtype=np.float32)
    if len(xs) == 0:
        return pos, neg

    duration = max(float(t1 - t0), 1.0)
    recency = (ts.astype(np.float32) - float(t0)) / duration
    recency = np.clip(recency, 0.0, 1.0)
    weights = recent_floor + (1.0 - recent_floor) * np.power(recency, recent_gamma)

    pos_mask = ps > 0
    neg_mask = ~pos_mask

    flat_pos = ys[pos_mask] * w + xs[pos_mask]
    flat_neg = ys[neg_mask] * w + xs[neg_mask]

    if flat_pos.size > 0:
        pos += np.bincount(flat_pos, weights=weights[pos_mask], minlength=h * w).reshape(h, w).astype(np.float32)
    if flat_neg.size > 0:
        neg += np.bincount(flat_neg, weights=weights[neg_mask], minlength=h * w).reshape(h, w).astype(np.float32)
    return pos, neg


def adaptive_update(
    state_pos: np.ndarray,
    state_neg: np.ndarray,
    pos: np.ndarray,
    neg: np.ndarray,
    dt_us: int,
    c_event: float,
    k0: float,
    decay_b: float,
    alpha: float,
    kmin: float,
    kmax: float,
    activity_tau: float,
    kernel: int,
    state_cap: float,
    inhibit_beta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    signed = pos - neg
    total = pos + neg

    act_local = box_filter(total, kernel)
    signed_local = box_filter(signed, kernel)

    act_norm = act_local / (act_local + max(activity_tau, 1e-6))
    consistency = np.abs(signed_local) / (act_local + 1e-6)
    reliability = np.clip(act_norm * consistency, 0.0, 1.0)

    k_eff = np.clip(k0 * (1.0 + alpha * (1.0 - reliability)), kmin, kmax)
    base = np.clip(1.0 - k_eff * float(dt_us), 1e-6, 1.0)
    decay = np.power(base, decay_b)

    decayed_pos = state_pos * decay
    decayed_neg = state_neg * decay

    raw_pos = decayed_pos + c_event * pos
    raw_neg = decayed_neg + c_event * neg

    if inhibit_beta > 0:
        comp_pos = np.maximum(raw_pos - inhibit_beta * raw_neg, 0.0)
        comp_neg = np.maximum(raw_neg - inhibit_beta * raw_pos, 0.0)
    else:
        comp_pos = raw_pos
        comp_neg = raw_neg

    new_pos = smooth_positive_saturate(comp_pos, vmax=state_cap)
    new_neg = smooth_positive_saturate(comp_neg, vmax=state_cap)
    return new_pos, new_neg


def pseudo_rgb_from_states(
    state_pos: np.ndarray,
    state_neg: np.ndarray,
    luma_gain: float,
    sum_scale: float,
    color_strength: float,
    gamma: float,
    dominance_boost: float,
) -> np.ndarray:
    dominant = np.maximum(state_pos, state_neg)
    total = state_pos + state_neg
    contrast = np.abs(state_pos - state_neg)

    structure = np.clip((1.0 - dominance_boost) * dominant + dominance_boost * contrast, 0.0, None)
    denom = max(np.log1p(luma_gain * max(sum_scale, 1e-6)), 1e-6)
    lum = np.log1p(luma_gain * structure) / denom
    lum = np.clip(lum, 0.0, 1.0)

    pol = (state_pos - state_neg) / (total + 1e-6)
    pol = np.clip(pol, -1.0, 1.0)
    pos_bias = np.clip(pol, 0.0, 1.0)
    neg_bias = np.clip(-pol, 0.0, 1.0)

    r = np.clip(lum + color_strength * pos_bias, 0.0, 1.0)
    g = lum
    b = np.clip(lum + color_strength * neg_bias, 0.0, 1.0)

    if gamma is not None and gamma > 0 and abs(gamma - 1.0) > 1e-6:
        r = np.power(r, gamma)
        g = np.power(g, gamma)
        b = np.power(b, gamma)

    rgb = np.stack([r, g, b], axis=-1)
    rgb = (255.0 * rgb).round().astype(np.uint8)
    return rgb


# -----------------------------------------------------------------------------
# Dataset traversal
# -----------------------------------------------------------------------------

def collect_videos(split_dir: Path, event_suffix: str) -> List[Tuple[Path, Path, str]]:
    items = []
    for h5_path in sorted(split_dir.glob(f"*{event_suffix}")):
        stem = h5_path.name[:-len(event_suffix)]
        bbox_path = split_dir / f"{stem}_bbox.npy"
        if not bbox_path.exists():
            print(f"[WARN] Missing bbox for {h5_path.name}, skip.")
            continue
        items.append((h5_path, bbox_path, stem))
    return items


def clean_split_outputs(images_dir: Path, label_json: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    label_json.parent.mkdir(parents=True, exist_ok=True)

    for p in images_dir.glob("*.png"):
        p.unlink()
    if label_json.exists():
        label_json.unlink()


def make_categories(config: DatasetConfig, seen_class_ids: Optional[Sequence[int]] = None) -> List[Dict[str, object]]:
    if config.class_names is not None:
        return [
            {"id": int(cid), "name": str(name)}
            for cid, name in sorted(config.class_names.items(), key=lambda item: item[0])
        ]

    if seen_class_ids is None:
        return []
    return [
        {"id": int(cid), "name": f"class_{int(cid)}"}
        for cid in sorted(seen_class_ids)
    ]


# -----------------------------------------------------------------------------
# Processing one split
# -----------------------------------------------------------------------------

def process_split(split: str, src_root: Path, out_root: Path, args, config: DatasetConfig) -> None:
    split_dir = src_root / split
    if not split_dir.exists():
        print(f"[WARN] Split not found: {split_dir}")
        return

    videos = collect_videos(split_dir, event_suffix=args.event_suffix)
    if len(videos) == 0:
        print(f"[WARN] No videos found in {split_dir} with suffix {args.event_suffix}")
        return

    images_dir = out_root / "images" / split
    labels_json = out_root / "labels" / f"{split}.json"

    if args.clean_output:
        clean_split_outputs(images_dir, labels_json)
    else:
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_json.parent.mkdir(parents=True, exist_ok=True)

    coco = {
        "videos": [],
        "images": [],
        "annotations": [],
        "categories": make_categories(config),
    }

    seen_class_ids = set()
    global_image_id = 1
    global_ann_id = 1
    global_video_id = 1

    for h5_path, bbox_path, video_stem in tqdm(videos, desc=f"Split={split} videos"):
        x, y, p, t = load_events_h5(str(h5_path))
        labels = load_bbox_npy(str(bbox_path))
        labels = mangle_labels(
            labels,
            split=split,
            config=config,
            sensor_w=args.sensor_w,
            sensor_h=args.sensor_h,
        )

        valid = (x >= 0) & (x < args.sensor_w) & (y >= 0) & (y < args.sensor_h)
        x, y, p, t = x[valid], y[valid], p[valid], t[valid]
        if len(t) == 0:
            print(f"[WARN] No valid events in {h5_path.name}, skip.")
            continue

        if config.class_names is None and len(labels) > 0:
            seen_class_ids.update(int(v) for v in np.unique(labels["class_id"]) if int(v) >= 0)

        video_id = global_video_id
        global_video_id += 1

        coco["videos"].append({
            "id": video_id,
            "name": video_stem,
            "file_name": h5_path.name,
        })

        state_pos = np.zeros((args.sensor_h, args.sensor_w), dtype=np.float32)
        state_neg = np.zeros((args.sensor_h, args.sensor_w), dtype=np.float32)

        start_t = int(t[0]) if args.start_us is None else int(args.start_us)
        end_t = int(t[-1]) if args.end_us is None else int(min(args.end_us, int(t[-1])))
        if end_t <= start_t:
            print(f"[WARN] Bad time range in {h5_path.name}, skip.")
            continue

        cur_evt = np.searchsorted(t, start_t, side="left")
        dt_us = int(args.window_us)

        frame_idx = 0
        cur_t0 = start_t
        while cur_t0 < end_t:
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

            cur_t1 = min(cur_t0 + dt_us, end_t)
            nxt_evt = np.searchsorted(t, cur_t1, side="left")

            xs = x[cur_evt:nxt_evt]
            ys = y[cur_evt:nxt_evt]
            ps = p[cur_evt:nxt_evt]
            ts = t[cur_evt:nxt_evt]

            pos, neg = accumulate_counts(
                xs, ys, ps, ts,
                cur_t0, cur_t1,
                args.sensor_h, args.sensor_w,
                recent_gamma=args.recent_gamma,
                recent_floor=args.recent_floor,
            )

            state_pos, state_neg = adaptive_update(
                state_pos=state_pos,
                state_neg=state_neg,
                pos=pos,
                neg=neg,
                dt_us=dt_us,
                c_event=args.c_event,
                k0=args.k0,
                decay_b=args.decay_b,
                alpha=args.alpha,
                kmin=args.kmin,
                kmax=args.kmax,
                activity_tau=args.activity_tau,
                kernel=args.kernel,
                state_cap=args.state_cap,
                inhibit_beta=args.inhibit_beta,
            )

            rgb = pseudo_rgb_from_states(
                state_pos=state_pos,
                state_neg=state_neg,
                luma_gain=args.luma_gain,
                sum_scale=args.sum_scale,
                color_strength=args.color_strength,
                gamma=args.gamma,
                dominance_boost=args.dominance_boost,
            )
            bgr = rgb[..., ::-1]

            frame_name = f"{video_stem}__frame_{frame_idx:05d}.png"
            frame_path = images_dir / frame_name
            if not cv2.imwrite(str(frame_path), bgr):
                raise IOError(f"Failed to write image: {frame_path}")

            frame_labels = labels[(labels["t"] >= cur_t0) & (labels["t"] < cur_t1)]
            if has_multiple_annotations_per_frame(frame_labels):
                frame_labels = cherry_pick_label_timestamps(frame_labels)

            image_id = global_image_id
            img_id = image_id
            global_image_id += 1

            coco["images"].append({
                "id": image_id,
                "img_id": img_id,
                "video_id": video_id,
                "frame_id": frame_idx,
                "file_name": f"images/{split}/{frame_name}",
                "width": args.sensor_w,
                "height": args.sensor_h,
                "timestamp_us": int(cur_t1),
            })

            for box in frame_labels:
                x0 = float(box["x"])
                y0 = float(box["y"])
                w = float(box["w"])
                h = float(box["h"])

                annotation = {
                    "id": global_ann_id,
                    "image_id": image_id,
                    "img_id": img_id,
                    "video_id": video_id,
                    "frame_id": frame_idx,
                    "category_id": int(box["class_id"]),
                    "bbox": [x0, y0, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                    "instance_id": int(box["track_id"]) if "track_id" in frame_labels.dtype.names else -1,
                }
                if config.include_box_timestamp:
                    annotation["box_timestamp_us"] = int(box["t"])

                coco["annotations"].append(annotation)
                global_ann_id += 1

            frame_idx += 1
            cur_evt = nxt_evt
            cur_t0 = cur_t1

    if config.class_names is None:
        coco["categories"] = make_categories(config, seen_class_ids=seen_class_ids)

    with open(labels_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"[DONE] dataset={config.name} split={split}")
    print(f"  images_dir: {images_dir}")
    print(f"  labels:     {labels_json}")
    print(f"  videos:     {len(coco['videos'])}")
    print(f"  images:     {len(coco['images'])}")
    print(f"  annos:      {len(coco['annotations'])}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Batch convert GEN1, GEN4, or ETRAM Prophesee datasets to HTA-RGB COCO-style layout."
    )
    ap.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_CONFIGS.keys()))
    ap.add_argument("--src_root", type=str, required=True, help="Original Prophesee dataset root.")
    ap.add_argument("--out_root", type=str, required=True, help="Output root for generated HTA-RGB data.")
    ap.add_argument("--sensor_w", type=int, default=None, help="Override dataset default sensor width.")
    ap.add_argument("--sensor_h", type=int, default=None, help="Override dataset default sensor height.")
    ap.add_argument("--event_suffix", type=str, default=None, help="Override dataset default event HDF5 suffix.")
    ap.add_argument("--splits", type=str, default="train,test", help="Comma-separated splits to process.")
    ap.add_argument("--window_us", type=int, default=50000)
    ap.add_argument("--start_us", type=int, default=None)
    ap.add_argument("--end_us", type=int, default=None)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--clean_output", action="store_true")

    ap.add_argument("--c_event", type=float, default=0.5)
    ap.add_argument("--k0", type=float, default=1.0e-5)
    ap.add_argument("--decay_b", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--kmin", type=float, default=5.0e-6)
    ap.add_argument("--kmax", type=float, default=2.0e-5)
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--activity_tau", type=float, default=4.0)
    ap.add_argument("--state_cap", type=float, default=8.0)
    ap.add_argument("--recent_gamma", type=float, default=3.5)
    ap.add_argument("--recent_floor", type=float, default=0.02)
    ap.add_argument("--inhibit_beta", type=float, default=0.30)

    ap.add_argument("--luma_gain", type=float, default=1.0)
    ap.add_argument("--sum_scale", type=float, default=8.0)
    ap.add_argument("--color_strength", type=float, default=0.16)
    ap.add_argument("--gamma", type=float, default=0.90)
    ap.add_argument("--dominance_boost", type=float, default=0.35)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    config = DATASET_CONFIGS[args.dataset]

    args.sensor_w = config.sensor_w if args.sensor_w is None else int(args.sensor_w)
    args.sensor_h = config.sensor_h if args.sensor_h is None else int(args.sensor_h)
    args.event_suffix = config.event_suffix if args.event_suffix is None else str(args.event_suffix)

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    print(f"[INFO] dataset={config.name} preprocess_family={config.preprocess_family}")
    print(f"[INFO] sensor={args.sensor_w}x{args.sensor_h} event_suffix={args.event_suffix}")
    print(f"[INFO] src_root={src_root}")
    print(f"[INFO] out_root={out_root}")

    splits = split_csv(args.splits)
    for split in splits:
        process_split(split, src_root, out_root, args, config=config)


if __name__ == "__main__":
    main()
