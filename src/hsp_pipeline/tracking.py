from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import GlobalInstance, SegmentRecord


@dataclass(frozen=True)
class MergeWeights:
    descriptor: float
    centroid: float
    iou: float
    size: float


def _bbox_iou_3d(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(inter))
    if inter_vol <= 0:
        return 0.0
    a_vol = float(np.prod(np.maximum(a_max - a_min, 0.0)))
    b_vol = float(np.prod(np.maximum(b_max - b_min, 0.0)))
    return inter_vol / max(a_vol + b_vol - inter_vol, 1e-9)


def _size_ratio_compat(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    a_vol = float(np.prod(np.maximum(a_max - a_min, 0.0)))
    b_vol = float(np.prod(np.maximum(b_max - b_min, 0.0)))
    if a_vol <= 0 or b_vol <= 0:
        return 0.0
    ratio = a_vol / b_vol
    if 0.3 <= ratio <= 3.0:
        return 1.0
    return 0.0


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def merge_score(
    seg: SegmentRecord,
    inst: GlobalInstance,
    weights: MergeWeights,
    max_centroid_distance: float,
    label_mismatch_penalty: float,
) -> float | None:
    d = float(np.linalg.norm(seg.centroid_xyz - inst.centroid))
    if d > max_centroid_distance:
        return None

    descriptor_score = (_cos(seg.descriptor, inst.descriptor) + 1.0) * 0.5
    centroid_score = max(0.0, 1.0 - (d / max_centroid_distance))
    iou = _bbox_iou_3d(seg.bbox_min, seg.bbox_max, inst.bbox_min, inst.bbox_max)
    size = _size_ratio_compat(seg.bbox_min, seg.bbox_max, inst.bbox_min, inst.bbox_max)

    score = (
        weights.descriptor * descriptor_score
        + weights.centroid * centroid_score
        + weights.iou * iou
        + weights.size * size
    )
    if seg.canonical_label != inst.canonical_label:
        score *= label_mismatch_penalty
    return score


class InstanceTracker:
    def __init__(
        self,
        ema_decay: float,
        merge_threshold: float,
        max_centroid_distance: float,
        label_mismatch_penalty: float,
        min_support_count: int,
        resweep_interval: int,
        skip_resweep: bool,
        weights: MergeWeights,
    ) -> None:
        self.ema_decay = ema_decay
        self.merge_threshold = merge_threshold
        self.max_centroid_distance = max_centroid_distance
        self.label_mismatch_penalty = label_mismatch_penalty
        self.min_support_count = min_support_count
        self.resweep_interval = resweep_interval
        self.skip_resweep = skip_resweep
        self.weights = weights

        self.instances: list[GlobalInstance] = []
        self._next_id = 1

    def _new_instance(self, seg: SegmentRecord) -> GlobalInstance:
        inst = GlobalInstance(
            instance_id=self._next_id,
            canonical_label=seg.canonical_label,
            points_3d=seg.points_3d.copy(),
            centroid=seg.centroid_xyz.copy(),
            bbox_min=seg.bbox_min.copy(),
            bbox_max=seg.bbox_max.copy(),
            descriptor=seg.descriptor.copy(),
        )
        inst.update(seg, ema_decay=self.ema_decay)
        self._next_id += 1
        return inst

    def _merge_or_create(self, seg: SegmentRecord) -> None:
        best_idx = None
        best_score = -1.0
        for i, inst in enumerate(self.instances):
            score = merge_score(
                seg,
                inst,
                weights=self.weights,
                max_centroid_distance=self.max_centroid_distance,
                label_mismatch_penalty=self.label_mismatch_penalty,
            )
            if score is None:
                continue
            if score > self.merge_threshold and score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            self.instances.append(self._new_instance(seg))
            return

        self.instances[best_idx].update(seg, ema_decay=self.ema_decay)

    def _resweep(self) -> None:
        changed = True
        while changed:
            changed = False
            for i in range(len(self.instances)):
                for j in range(i + 1, len(self.instances)):
                    a = self.instances[i]
                    b = self.instances[j]
                    pseudo = SegmentRecord(
                        frame_ts=b.last_ts,
                        sec=0,
                        nsec=0,
                        label=b.canonical_label,
                        canonical_label=b.canonical_label,
                        sam3_score=b.sam3_stats.mean,
                        descriptor=b.descriptor,
                        points_3d=b.points_3d,
                        mast3r_conf=np.array([b.mast3r_conf_mean], dtype=np.float32),
                        centroid_xyz=b.centroid,
                        bbox_min=b.bbox_min,
                        bbox_max=b.bbox_max,
                        is_structural=False,
                    )
                    score = merge_score(
                        pseudo,
                        a,
                        weights=self.weights,
                        max_centroid_distance=self.max_centroid_distance,
                        label_mismatch_penalty=self.label_mismatch_penalty,
                    )
                    if score is not None and score > self.merge_threshold:
                        a.points_3d = np.concatenate([a.points_3d, b.points_3d], axis=0)
                        a.centroid = a.points_3d.mean(axis=0)
                        a.bbox_min = np.minimum(a.bbox_min, b.bbox_min)
                        a.bbox_max = np.maximum(a.bbox_max, b.bbox_max)
                        a.descriptor = self.ema_decay * a.descriptor + (1.0 - self.ema_decay) * b.descriptor
                        a.sam3_stats.merge(b.sam3_stats)
                        total_conf = (
                            a.mast3r_conf_mean * a.mast3r_conf_count
                            + b.mast3r_conf_mean * b.mast3r_conf_count
                        )
                        a.mast3r_conf_count += b.mast3r_conf_count
                        if a.mast3r_conf_count > 0:
                            a.mast3r_conf_mean = total_conf / a.mast3r_conf_count
                        a.support_count += b.support_count
                        a.last_ts = max(a.last_ts, b.last_ts)
                        a.first_ts = min(a.first_ts, b.first_ts)
                        a.source_frame_list.extend(b.source_frame_list)
                        self.instances.pop(j)
                        changed = True
                        break
                if changed:
                    break

    def update_frame(self, frame_segments: list[SegmentRecord], frame_idx: int) -> None:
        for seg in frame_segments:
            if seg.is_structural:
                continue
            self._merge_or_create(seg)

        if not self.skip_resweep and self.resweep_interval > 0 and (frame_idx + 1) % self.resweep_interval == 0:
            self._resweep()

    def final_split(self) -> tuple[list[GlobalInstance], list[GlobalInstance]]:
        keep, low = [], []
        for inst in self.instances:
            (keep if inst.support_count >= self.min_support_count else low).append(inst)
        return keep, low
