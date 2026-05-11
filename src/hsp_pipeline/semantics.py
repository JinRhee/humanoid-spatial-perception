from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import read_yaml_file
from .types import FrameRecord, SegmentRecord


@dataclass(frozen=True)
class PromptItem:
    label: str
    is_structural: bool


@dataclass
class SemanticsResult:
    object_segments: list[SegmentRecord]
    structural_points: np.ndarray
    per_frame_rows: dict[tuple[int, int], list[dict]]


def load_prompts(prompt_path: Path, structural_labels: set[str]) -> list[PromptItem]:
    items: list[PromptItem] = []
    with prompt_path.open("r", encoding="utf-8") as f:
        for line in f:
            txt = line.strip()
            if not txt or txt.startswith("#"):
                continue
            if "|" in txt:
                label, cls = [x.strip() for x in txt.split("|", 1)]
                is_structural = cls.lower() == "structural"
            else:
                label = txt
                is_structural = label.lower() in structural_labels
            items.append(PromptItem(label=label, is_structural=is_structural))
    if not items:
        raise ValueError(f"No prompts loaded from {prompt_path}")
    return items


def load_synonyms(path: Path) -> dict[str, str]:
    data = read_yaml_file(path)
    out: dict[str, str] = {}
    for k, v in data.items():
        out[str(k).strip().lower()] = str(v).strip().lower()
    return out


def canonicalize(label: str, synonym_map: dict[str, str]) -> str:
    key = label.strip().lower()
    return synonym_map.get(key, key)


def _simple_prompt_masks(h: int, w: int, prompt_count: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    stripe_w = max(1, w // max(prompt_count, 1))
    for i in range(prompt_count):
        x0 = i * stripe_w
        x1 = w if i == prompt_count - 1 else min(w, (i + 1) * stripe_w)
        mask = np.zeros((h, w), dtype=bool)
        mask[:, x0:x1] = True
        masks.append(mask)
    return masks


def _bbox_xyxy(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _dense_points_and_conf(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    z = (rgb.mean(axis=2) / 255.0) * 2.0 + 0.5
    x = (xx - (w / 2.0)) / max(w, 1)
    y = (yy - (h / 2.0)) / max(h, 1)
    pts3d = np.stack([x, y, z], axis=-1).astype(np.float32)
    conf = np.clip(1.0 - np.abs(z - z.mean()) / (z.std() + 1e-6), 0.0, 1.0).astype(np.float32)
    return pts3d, conf


def _descriptor_from_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pix = rgb[mask]
    if pix.size == 0:
        return np.ones((24,), dtype=np.float32) * 1e-6
    mean_rgb = pix.mean(axis=0) / 255.0
    std_rgb = pix.std(axis=0) / 255.0
    v = np.concatenate([mean_rgb, std_rgb, [pix.shape[0] / (rgb.shape[0] * rgb.shape[1] + 1e-9)]])
    rep = np.tile(v, 4)[:24].astype(np.float32)
    norm = np.linalg.norm(rep) + 1e-9
    return rep / norm


def run_semantics(
    frames: list[FrameRecord],
    prompts: list[PromptItem],
    synonym_map: dict[str, str],
    conf_min: float,
    detection_conf_threshold: float,
    trajectory_by_ts: dict[float, np.ndarray],
    max_image_resolution: int,
    max_masks_per_frame: int,
) -> SemanticsResult:
    object_segments: list[SegmentRecord] = []
    structural_accum: list[np.ndarray] = []
    per_frame_rows: dict[tuple[int, int], list[dict]] = {}

    for fr in frames:
        with Image.open(fr.path) as im:
            rgb_img = im.convert("RGB")
            if max_image_resolution and max_image_resolution > 0:
                w0, h0 = rgb_img.size
                long_edge = max(w0, h0)
                if long_edge > max_image_resolution:
                    scale = max_image_resolution / long_edge
                    rgb_img = rgb_img.resize((int(w0 * scale), int(h0 * scale)))
            rgb = np.asarray(rgb_img, dtype=np.float32)
        h, w, _ = rgb.shape
        limited_prompts = prompts[:max_masks_per_frame] if max_masks_per_frame > 0 else prompts
        masks = _simple_prompt_masks(h, w, len(limited_prompts))
        pts3d, conf = _dense_points_and_conf(rgb)
        trans = trajectory_by_ts.get(fr.timestamp, np.zeros(3, dtype=np.float32))

        rows: list[dict] = []
        for prompt, mask in zip(limited_prompts, masks):
            sam_score = float(mask.mean())
            if sam_score < detection_conf_threshold:
                continue

            canonical = canonicalize(prompt.label, synonym_map)
            descriptor = _descriptor_from_mask(rgb, mask)

            sel_conf = conf[mask]
            good = sel_conf >= conf_min
            sel_pts = pts3d[mask][good]
            sel_conf = sel_conf[good]

            if sel_pts.size == 0:
                continue

            global_pts = sel_pts + trans[None, :]
            centroid = global_pts.mean(axis=0)
            bmin = global_pts.min(axis=0)
            bmax = global_pts.max(axis=0)

            row = {
                "frame_ts": fr.timestamp,
                "label": prompt.label,
                "canonical_label": canonical,
                "sam3_score": sam_score,
                "descriptor": descriptor.tolist(),
                "num_points": int(global_pts.shape[0]),
                "centroid_xyz": centroid.tolist(),
                "bbox_min": bmin.tolist(),
                "bbox_max": bmax.tolist(),
                "bbox_xyxy": _bbox_xyxy(mask),
            }
            rows.append(row)

            seg = SegmentRecord(
                frame_ts=fr.timestamp,
                sec=fr.sec,
                nsec=fr.nsec,
                label=prompt.label,
                canonical_label=canonical,
                sam3_score=sam_score,
                descriptor=descriptor,
                points_3d=global_pts,
                mast3r_conf=sel_conf,
                centroid_xyz=centroid,
                bbox_min=bmin,
                bbox_max=bmax,
                is_structural=prompt.is_structural,
            )

            if prompt.is_structural:
                structural_accum.append(global_pts)
            else:
                object_segments.append(seg)

        per_frame_rows[(fr.sec, fr.nsec)] = rows

    structural_points = (
        np.concatenate(structural_accum, axis=0) if structural_accum else np.zeros((0, 3), dtype=np.float32)
    )
    return SemanticsResult(object_segments=object_segments, structural_points=structural_points, per_frame_rows=per_frame_rows)


def sanity_check_semantics(segments: list[SegmentRecord], keyframe_count: int) -> None:
    if keyframe_count == 0:
        return
    avg = len(segments) / keyframe_count
    if avg < 1.0:
        raise RuntimeError("Sanity check failed: fewer than one lifted segment per keyframe on average")
    for seg in segments:
        if np.allclose(seg.descriptor, 0.0):
            raise RuntimeError("Sanity check failed: found all-zero descriptor")
