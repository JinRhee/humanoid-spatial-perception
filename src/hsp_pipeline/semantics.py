from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import Sam3Adapter, Sam3MaskResult, SegMASt3RAdapter
from .config import read_yaml_file
from .spatial import transform_points
from .types import BackboneOutput, FrameRecord, Mast3rFeaturesV1, Mast3rFeaturesV2, SegmentRecord


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


def _bbox_xyxy(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def _select_features(
    frame_out,
    version: str,
) -> Mast3rFeaturesV1 | Mast3rFeaturesV2:
    if version == "v1":
        return frame_out.features_v1
    if version == "v2":
        return frame_out.features_v2
    raise ValueError(f"Unknown SegMASt3R feature version '{version}'")


def _mask_to_points_and_conf(frame_out, mask: np.ndarray, conf_min: float) -> tuple[np.ndarray, np.ndarray]:
    if mask.shape != frame_out.pointmap.confidence.shape:
        raise ValueError("SAM3 mask shape does not match pointmap resolution")
    conf = frame_out.pointmap.confidence[mask]
    pts = frame_out.pointmap.points[mask]
    good = conf >= conf_min
    return pts[good], conf[good]


def run_semantics(
    frames: list[FrameRecord],
    backbone: BackboneOutput,
    prompts: list[PromptItem],
    synonym_map: dict[str, str],
    conf_min: float,
    detection_conf_threshold: float,
    pose_matrices: dict[float, np.ndarray],
    max_masks_per_frame: int,
    sam3_cfg: dict[str, Any],
    segmast3r_cfg: dict[str, Any],
    checkpoints: dict[str, Any],
) -> SemanticsResult:
    object_segments: list[SegmentRecord] = []
    structural_accum: list[np.ndarray] = []
    per_frame_rows: dict[tuple[int, int], list[dict]] = {}

    sam3 = Sam3Adapter(sam3_cfg, checkpoints=checkpoints)
    segmast3r = SegMASt3RAdapter(segmast3r_cfg, checkpoints=checkpoints)
    feature_version = str(segmast3r_cfg.get("feature_version", "v2"))

    for fr in frames:
        frame_out = backbone.frames.get((fr.sec, fr.nsec))
        if frame_out is None:
            raise RuntimeError(f"Missing backbone output for frame {fr.path}")
        rgb = frame_out.rgb
        limited_prompts = prompts[:max_masks_per_frame] if max_masks_per_frame > 0 else prompts
        labels = [p.label for p in limited_prompts]
        masks = sam3.predict(rgb, labels, max_masks_per_frame)
        pose = pose_matrices.get(fr.timestamp)
        if pose is None:
            raise RuntimeError(f"Missing pose for frame {fr.path}")
        features = _select_features(frame_out, feature_version)

        rows: list[dict] = []
        for prompt, mask_result in zip(limited_prompts, masks):
            if isinstance(mask_result, Sam3MaskResult):
                mask = mask_result.mask
                sam_score = float(mask_result.score)
            else:
                mask = mask_result
                sam_score = float(mask.mean())
            if sam_score < detection_conf_threshold:
                continue

            canonical = canonicalize(prompt.label, synonym_map)
            descriptor = segmast3r.encode_mask(features, mask)
            sel_pts, sel_conf = _mask_to_points_and_conf(frame_out, mask, conf_min)
            if sel_pts.size == 0:
                continue

            global_pts = transform_points(sel_pts, pose)
            centroid = global_pts.mean(axis=0)
            bmin = global_pts.min(axis=0)
            bmax = global_pts.max(axis=0)

            row = {
                "frame_ts": fr.timestamp,
                "label": prompt.label,
                "canonical_label": canonical,
                "sam3_score": sam_score,
                "descriptor": descriptor.tolist(),
                "point_confidence_mean": float(sel_conf.mean()) if sel_conf.size > 0 else 0.0,
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


def sanity_check_semantics(segments: list[SegmentRecord], keyframe_count: int, max_masks_per_frame: int) -> None:
    if keyframe_count == 0:
        return
    if max_masks_per_frame > 0:
        max_expected = keyframe_count * max_masks_per_frame
        if len(segments) > max_expected:
            raise RuntimeError("Sanity check failed: segment count exceeds max masks per frame")
    if not segments:
        print("[warn] No object segments produced; check prompts or SAM3 thresholds.")
    for seg in segments:
        if not np.all(np.isfinite(seg.descriptor)) or np.linalg.norm(seg.descriptor) < 1e-9:
            raise RuntimeError("Sanity check failed: found invalid or near-zero descriptor")
        if seg.points_3d.size == 0:
            raise RuntimeError("Sanity check failed: segment has zero points")
        if seg.mast3r_conf.size and (np.any(seg.mast3r_conf < 0.0) or np.any(seg.mast3r_conf > 1.0)):
            raise RuntimeError("Sanity check failed: point confidence out of range")
        if not (0.0 <= seg.sam3_score <= 1.0):
            raise RuntimeError("Sanity check failed: SAM3 score out of range")
