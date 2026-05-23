from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .utils_spatial import transform_points


EPS = 1e-8


@dataclass
class SegmentRecord:
    mask_id: int
    frame_index: int
    descriptor: np.ndarray
    points_world: np.ndarray
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    label: str | None
    score: float
    instance_id: int | None = None
    is_matched: bool = False
    frame_ts: float | None = None

    @property
    def size(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


@dataclass
class SegmentationStore:
    segments_by_frame: dict[int, list[SegmentRecord]] = field(default_factory=dict)
    matches_by_pair: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)

    def add_segments(self, frame_index: int, segments: list[SegmentRecord]) -> None:
        self.segments_by_frame[frame_index] = segments

    def add_match_result(self, frame_a: int, frame_b: int, match_result: np.ndarray) -> None:
        self.matches_by_pair[(frame_a, frame_b)] = np.asarray(match_result)

    def get_segments(self, frame_index: int) -> list[SegmentRecord]:
        return self.segments_by_frame.get(frame_index, [])

    def get_match_result(self, frame_a: int, frame_b: int) -> np.ndarray | None:
        return self.matches_by_pair.get((frame_a, frame_b))


@dataclass(frozen=True)
class MergeWeights:
    descriptor: float
    centroid: float
    overlap: float
    size: float


@dataclass
class InstanceState:
    instance_id: int
    descriptor: np.ndarray
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    support_count: int
    last_seen: int
    label: str | None
    score: float
    points_3d: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    keyframes: list[int] = field(default_factory=list)
    keyframe_masks: dict[int, np.ndarray] = field(default_factory=dict)
    is_active: bool = True

    @property
    def size(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


@dataclass
class CandidateState:
    instance_id: int
    descriptor: np.ndarray
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    label: str | None
    score: float
    frames_appeared: list[int] = field(default_factory=list)
    points_3d: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    keyframe_masks: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def size(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


def _to_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


def _centroid_similarity(a: np.ndarray, b: np.ndarray, max_dist: float) -> float:
    return max(0.0, 1.0 - float(np.linalg.norm(a - b)) / max_dist)


def _size_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return max(0.0, 1.0 - np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + EPS))


def _bbox_max_overlap(min_a, max_a, min_b, max_b) -> float:
    """max(I/vol_a, I/vol_b) — high when either bbox is largely contained in the other."""
    min_a, max_a = np.asarray(min_a, dtype=np.float32), np.asarray(max_a, dtype=np.float32)
    min_b, max_b = np.asarray(min_b, dtype=np.float32), np.asarray(max_b, dtype=np.float32)
    inter = np.maximum(0.0, np.minimum(max_a, max_b) - np.maximum(min_a, min_b))
    inter_vol = float(np.prod(inter))
    vol_a = float(np.prod(np.maximum(0.0, max_a - min_a)))
    vol_b = float(np.prod(np.maximum(0.0, max_b - min_b)))
    if vol_a < EPS and vol_b < EPS:
        return 0.0
    return max(inter_vol / (vol_a + EPS), inter_vol / (vol_b + EPS))


BOX_EDGES = np.array(
    [[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]],
    dtype=np.int32,
)


def bbox_iou(min_a, max_a, min_b, max_b) -> float:
    return _bbox_max_overlap(min_a, max_a, min_b, max_b)


def bbox_corners(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    mn = np.asarray(bbox_min, dtype=np.float32)
    mx = np.asarray(bbox_max, dtype=np.float32)
    return np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mn[0], mx[1], mn[2]], [mx[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mn[0], mx[1], mx[2]], [mx[0], mx[1], mx[2]],
    ], dtype=np.float32)


def instance_rgba(instance_id: int, alpha: float = 1.0) -> np.ndarray:
    seed = int(np.uint32(instance_id) * np.uint32(2654435761))
    rng = np.random.default_rng(seed)
    rgb = rng.uniform(0.35, 0.95, size=3).astype(np.float32)
    return np.concatenate([rgb, np.array([alpha], dtype=np.float32)])


def build_segment_records(
    frame_index: int,
    masks,
    points_3d,
    conf,
    descriptors,
    pose_world: np.ndarray | None = None,
    frame_ts: float | None = None,
    labels: list | None = None,
    score: float = 1.0,
) -> list[SegmentRecord]:
    masks_np = _to_numpy(masks)
    points_np = _to_numpy(points_3d)
    desc_np = _to_numpy(descriptors)

    if desc_np.ndim == 3:
        desc_np = desc_np[0]
    if masks_np.ndim == 4:
        masks_np = masks_np[0]

    segments: list[SegmentRecord] = []
    for mask_id in range(masks_np.shape[0]):
        mask = masks_np[mask_id] > 0
        if mask.sum() == 0:
            continue
        pts = points_np[mask]
        if pose_world is not None:
            pts = transform_points(pts, pose_world)
        seg_label = labels[mask_id] if labels is not None and mask_id < len(labels) else None
        segments.append(SegmentRecord(
            mask_id=mask_id,
            frame_index=frame_index,
            descriptor=desc_np[:, mask_id].astype(np.float32, copy=False),
            points_world=pts.astype(np.float32, copy=False),
            centroid=pts.mean(axis=0).astype(np.float32, copy=False),
            bbox_min=pts.min(axis=0).astype(np.float32, copy=False),
            bbox_max=pts.max(axis=0).astype(np.float32, copy=False),
            label=seg_label,
            score=score,
            frame_ts=frame_ts,
        ))
    return segments


class InstanceTracker:
    def __init__(
        self,
        merge_weights: MergeWeights | None = None,
        merge_threshold: float = 0.5,
        reattach_threshold: float = 0.4,
        max_centroid_distance: float = 2.0,
        label_mismatch_penalty: float = 0.5,
        ema_decay: float = 0.7,
        min_support_count: int = 3,
        candidate_promotion_threshold: int = 2,
        candidate_window_size: int = 3,
        skip_resweep: bool = False,
        label_filter: bool = False,
        **kwargs,
    ) -> None:
        self.merge_weights = merge_weights or MergeWeights(descriptor=0.5, centroid=0.3, iou=0.1, size=0.1)
        self.merge_threshold = merge_threshold
        self.reattach_threshold = reattach_threshold
        self.max_centroid_distance = max_centroid_distance
        self.label_mismatch_penalty = label_mismatch_penalty
        self.ema_decay = ema_decay
        self.min_support_count = min_support_count
        self.candidate_promotion_threshold = candidate_promotion_threshold
        self.candidate_window_size = candidate_window_size
        self.skip_resweep = skip_resweep
        self.label_filter = label_filter

        self.valid_instances: dict[int, InstanceState] = {}
        self.candidate_instances: dict[int, CandidateState] = {}
        self._next_instance_id = 0
        self._prev_segments: list[SegmentRecord] = []

    def update_frame(
        self,
        segments: list[SegmentRecord],
        frame_index: int,
        match_result: np.ndarray | None = None,
        masks: np.ndarray | None = None,  # (N, H, W)
    ) -> None:
        # Clear IDs that were merged away by a resweep since the last frame
        live = self.valid_instances.keys() | self.candidate_instances.keys()
        for seg in self._prev_segments:
            if seg.instance_id not in live:
                seg.instance_id = None

        # --- Phase 1: propagate IDs from prev frame via feature matcher ---
        assignments: dict[int, int] = {}
        n_match_no_match = 0
        n_match_no_prev_id = 0
        n_match_duplicate = 0
        if match_result is not None and self._prev_segments:
            for prev_idx, curr_idx in enumerate(np.asarray(match_result).tolist()):
                curr_idx = int(curr_idx)
                if curr_idx < 0:
                    n_match_no_match += 1
                    continue
                prev_seg = self._prev_segments[prev_idx]
                if prev_seg.instance_id is None:
                    n_match_no_prev_id += 1
                    continue
                if curr_idx not in assignments:
                    assignments[curr_idx] = prev_seg.instance_id
                else:
                    n_match_duplicate += 1

        def get_mask(seg: SegmentRecord) -> np.ndarray | None:
            if masks is None:
                return None
            return _to_numpy(masks[seg.mask_id]).astype(np.bool_)

        matched_valid_ids: set[int] = set()
        matched_segs: list[SegmentRecord] = []

        n_skip_label = 0
        n_matched_valid = 0
        n_reattached = 0
        n_matched_candidate = 0
        n_new_candidate = 0

        for idx, seg in enumerate(segments):
            if self.label_filter and seg.label is None:
                n_skip_label += 1
                continue
            mask = get_mask(seg)
            instance_id = assignments.get(idx)

            # --- Phase 2: fallback — geometry-only score against valid instances and candidates ---
            if instance_id not in self.valid_instances and instance_id not in self.candidate_instances:
                best_id, best_score = None, -1.0
                for iid, inst in self.valid_instances.items():
                    s = self._score_pair_geometry(inst, seg)
                    if s > best_score:
                        best_score, best_id = s, iid
                for iid, cand in self.candidate_instances.items():
                    s = self._score_pair_geometry(cand, seg)
                    if s > best_score:
                        best_score, best_id = s, iid
                if best_score >= self.reattach_threshold:
                    instance_id = best_id
                    n_reattached += 1

            if instance_id in self.valid_instances:
                matched_valid_ids.add(instance_id)
                matched_segs.append(seg)
                self._update_instance(self.valid_instances[instance_id], seg, frame_index, mask)
                seg.instance_id = instance_id
                seg.is_matched = True
                n_matched_valid += 1
            elif instance_id in self.candidate_instances:
                self._update_candidate(self.candidate_instances[instance_id], seg, frame_index, mask)
                seg.instance_id = instance_id
                seg.is_matched = True
                n_matched_candidate += 1
            else:
                seg.instance_id = self._create_candidate(seg, frame_index, mask)
                n_new_candidate += 1

        n_total = len(segments)
        if match_result is not None:
            mr = np.asarray(match_result)
            print(f"[match_result] shape={mr.shape} min={mr.min()} max={mr.max()} "
                  f"positive={int((mr >= 0).sum())} negative={int((mr < 0).sum())} "
                  f"values={mr.tolist()}")
        print(
            f"[update_frame] frame={frame_index} segs={n_total} "
            f"| matches: no_match={n_match_no_match} no_prev_id={n_match_no_prev_id} dup={n_match_duplicate} assigned={len(assignments)} "
            f"| label_skip={n_skip_label} "
            f"| -> valid={n_matched_valid} reattached={n_reattached} candidate={n_matched_candidate} new={n_new_candidate} "
            f"| pool: valid={len(self.valid_instances)} candidates={len(self.candidate_instances)}"
        )

        for iid in self.valid_instances:
            if iid not in matched_valid_ids:
                inst = self.valid_instances[iid]
                # Containment veto: if this instance's centroid lies inside any matched
                # segment's 3D bbox, assume undersegmentation rather than true loss.
                absorbed = any(
                    np.all(s.bbox_min <= inst.centroid) and np.all(inst.centroid <= s.bbox_max)
                    for s in matched_segs
                )
                inst.is_active = absorbed

        self._promote_candidates(frame_index)
        self._prune_broken_candidates(frame_index)
        self._prev_segments = segments

    def _create_candidate(self, seg: SegmentRecord, frame_index: int, mask: np.ndarray | None = None) -> int:
        instance_id = self._next_instance_id
        self._next_instance_id += 1
        self.candidate_instances[instance_id] = CandidateState(
            instance_id=instance_id,
            descriptor=seg.descriptor.copy(),
            centroid=seg.centroid.copy(),
            bbox_min=seg.bbox_min.copy(),
            bbox_max=seg.bbox_max.copy(),
            label=seg.label,
            score=seg.score,
            frames_appeared=[frame_index],
            points_3d=seg.points_world.copy(),
            keyframe_masks={frame_index: mask},
        )
        return instance_id

    def _update_candidate(self, cand: CandidateState, seg: SegmentRecord, frame_index: int, mask: np.ndarray | None = None) -> None:
        alpha = self.ema_decay
        cand.descriptor = alpha * cand.descriptor + (1.0 - alpha) * seg.descriptor
        cand.centroid = alpha * cand.centroid + (1.0 - alpha) * seg.centroid
        cand.bbox_min = np.minimum(cand.bbox_min, seg.bbox_min)
        cand.bbox_max = np.maximum(cand.bbox_max, seg.bbox_max)
        cand.points_3d = np.concatenate([cand.points_3d, seg.points_world], axis=0)
        cand.keyframe_masks[frame_index] = mask
        cand.frames_appeared.append(frame_index)
        if seg.label and (cand.label is None or seg.score >= cand.score):
            cand.label = seg.label
        cand.score = max(cand.score, seg.score)

    def _update_instance(self, inst: InstanceState, seg: SegmentRecord, frame_index: int, mask: np.ndarray | None = None) -> None:
        alpha = self.ema_decay
        inst.descriptor = alpha * inst.descriptor + (1.0 - alpha) * seg.descriptor
        inst.centroid = alpha * inst.centroid + (1.0 - alpha) * seg.centroid
        inst.bbox_min = np.minimum(inst.bbox_min, seg.bbox_min)
        inst.bbox_max = np.maximum(inst.bbox_max, seg.bbox_max)
        inst.support_count += 1
        inst.last_seen = frame_index
        inst.is_active = True
        inst.points_3d = np.concatenate([inst.points_3d, seg.points_world], axis=0)
        inst.keyframes.append(frame_index)
        inst.keyframe_masks[frame_index] = mask
        if seg.label and (inst.label is None or seg.score >= inst.score):
            inst.label = seg.label
        inst.score = max(inst.score, seg.score)

    def _promote_candidates(self, frame_index: int) -> None:
        window_start = frame_index - self.candidate_window_size
        to_promote = [
            iid for iid, cand in self.candidate_instances.items()
            if sum(1 for f in cand.frames_appeared if f >= window_start) >= self.candidate_promotion_threshold
        ]
        for iid in to_promote:
            cand = self.candidate_instances.pop(iid)
            self.valid_instances[iid] = InstanceState(
                instance_id=cand.instance_id,
                descriptor=cand.descriptor.copy(),
                centroid=cand.centroid.copy(),
                bbox_min=cand.bbox_min.copy(),
                bbox_max=cand.bbox_max.copy(),
                support_count=len(cand.frames_appeared),
                last_seen=frame_index,
                label=cand.label,
                score=cand.score,
                points_3d=cand.points_3d.copy(),
                keyframes=cand.frames_appeared.copy(),
                keyframe_masks=cand.keyframe_masks.copy(),
            )
        if to_promote:
            print(f"[promote] {len(to_promote)} candidate(s) promoted: {to_promote}")

    def _prune_broken_candidates(self, frame_index: int) -> None:
        stale = [iid for iid, cand in self.candidate_instances.items()
                 if cand.frames_appeared[-1] < frame_index - self.candidate_window_size]
        for iid in stale:
            del self.candidate_instances[iid]
        if stale:
            print(f"[prune] {len(stale)} stale candidate(s) pruned: {stale}")

    def _score_pair(self, a, b) -> float:
        """Full score including descriptor — for consecutive matching and resweep."""
        w = self.merge_weights
        score = (
            w.descriptor * _cosine_similarity(a.descriptor, b.descriptor)
            + w.centroid  * _centroid_similarity(a.centroid, b.centroid, self.max_centroid_distance)
            + w.overlap   * _bbox_max_overlap(a.bbox_min, a.bbox_max, b.bbox_min, b.bbox_max)
            + w.size      * _size_similarity(a.size, b.size)
        )
        if a.label and b.label and a.label != b.label:
            score *= self.label_mismatch_penalty
        return score

    def _score_pair_geometry(self, a, b) -> float:
        """Geometry-only score (no descriptor) — for cross-gap reattachment."""
        w = self.merge_weights
        total_geo = w.centroid + w.overlap + w.size
        if total_geo < EPS:
            return 0.0
        score = (
            w.centroid * _centroid_similarity(a.centroid, b.centroid, self.max_centroid_distance)
            + w.overlap * _bbox_max_overlap(a.bbox_min, a.bbox_max, b.bbox_min, b.bbox_max)
            + w.size    * _size_similarity(a.size, b.size)
        ) / total_geo
        if a.label and b.label and a.label != b.label:
            score *= self.label_mismatch_penalty
        return score

    def _score_instance_pair(self, inst_a: InstanceState, inst_b: InstanceState) -> float:
        return self._score_pair(inst_a, inst_b)

    def _merge_instances(self, instances: Iterable[InstanceState]) -> InstanceState:
        inst_list = list(instances)
        weights = np.array([inst.support_count for inst in inst_list], dtype=np.float32)
        weights /= weights.sum()

        def weighted_avg(vals):
            return (weights[:, None] * np.stack(list(vals))).sum(axis=0)

        all_keyframes: list[int] = []
        all_keyframe_masks: dict[int, np.ndarray] = {}
        for inst in inst_list:
            all_keyframes.extend(inst.keyframes)
            all_keyframe_masks.update(inst.keyframe_masks)

        label, score = self._merge_labels(inst_list)

        return InstanceState(
            instance_id=min(inst.instance_id for inst in inst_list),
            descriptor=weighted_avg(inst.descriptor for inst in inst_list).astype(np.float32),
            centroid=weighted_avg(inst.centroid for inst in inst_list).astype(np.float32),
            bbox_min=np.min([inst.bbox_min for inst in inst_list], axis=0).astype(np.float32),
            bbox_max=np.max([inst.bbox_max for inst in inst_list], axis=0).astype(np.float32),
            support_count=sum(inst.support_count for inst in inst_list),
            last_seen=max(inst.last_seen for inst in inst_list),
            label=label,
            score=score,
            points_3d=np.concatenate([inst.points_3d for inst in inst_list]).astype(np.float32),
            keyframes=all_keyframes,
            keyframe_masks=all_keyframe_masks,
        )

    def _merge_labels(self, instances: Iterable[InstanceState]) -> tuple[str | None, float]:
        best_label, best_score = None, -1.0
        for inst in instances:
            if inst.label is None:
                continue
            s = inst.score * inst.support_count
            if s > best_score:
                best_score, best_label = s, inst.label
        return best_label, max(best_score, 0.0)

    def resweep(self) -> None:
        if len(self.valid_instances) < 2:
            return
        ids = list(self.valid_instances.keys())
        parent = {iid: iid for iid in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, id_a in enumerate(ids):
            inst_a = self.valid_instances[id_a]
            for id_b in ids[i + 1:]:
                inst_b = self.valid_instances[id_b]
                if float(np.linalg.norm(inst_a.centroid - inst_b.centroid)) < 0.3:
                    union(id_a, id_b)
                elif self._score_pair(inst_a, inst_b) >= self.merge_threshold:
                    union(id_a, id_b)

        clusters: dict[int, list[int]] = {}
        for iid in ids:
            clusters.setdefault(find(iid), []).append(iid)

        merged: dict[int, InstanceState] = {}
        for cluster in clusters.values():
            if len(cluster) == 1:
                merged[cluster[0]] = self.valid_instances[cluster[0]]
            else:
                inst = self._merge_instances(self.valid_instances[iid] for iid in cluster)
                merged[inst.instance_id] = inst
        self.valid_instances = merged

    def final_resweep(self, centroid_distance_threshold: float = 0.3) -> None:
        if len(self.valid_instances) < 2:
            return
        ids = list(self.valid_instances.keys())
        parent = {iid: iid for iid in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, id_a in enumerate(ids):
            inst_a = self.valid_instances[id_a]
            for id_b in ids[i + 1:]:
                inst_b = self.valid_instances[id_b]
                if float(np.linalg.norm(inst_a.centroid - inst_b.centroid)) < centroid_distance_threshold:
                    union(id_a, id_b)

        clusters: dict[int, list[int]] = {}
        for iid in ids:
            clusters.setdefault(find(iid), []).append(iid)

        merged: dict[int, InstanceState] = {}
        for cluster in clusters.values():
            if len(cluster) == 1:
                merged[cluster[0]] = self.valid_instances[cluster[0]]
            else:
                inst = self._merge_instances(self.valid_instances[iid] for iid in cluster)
                merged[inst.instance_id] = inst
        self.valid_instances = merged
        print(f"======final_resweep(): {len(self.valid_instances) - len(ids)}")

    def final_split(self) -> tuple[list[InstanceState], list[InstanceState]]:
        if not self.skip_resweep:
            print("FINAL RESWEEP")
            self.final_resweep()
        instances = list(self.valid_instances.values())
        keep = [inst for inst in instances if inst.support_count >= self.min_support_count]
        low  = [inst for inst in instances if inst.support_count <  self.min_support_count]
        return keep, low

    def get_valid_instances_at_keyframe(self, keyframe_idx: int) -> list[InstanceState]:
        return [inst for inst in self.valid_instances.values() if keyframe_idx in inst.keyframes]

    def get_candidate_instances_at_keyframe(self, keyframe_idx: int) -> list[CandidateState]:
        return [cand for cand in self.candidate_instances.values() if keyframe_idx in cand.frames_appeared]

    def get_all_instances_at_keyframe(self, keyframe_idx: int) -> tuple[list[InstanceState], list[CandidateState]]:
        return self.get_valid_instances_at_keyframe(keyframe_idx), self.get_candidate_instances_at_keyframe(keyframe_idx)

    def summaries(self) -> list[dict]:
        return [
            {
                "instance_id": inst.instance_id,
                "support_count": inst.support_count,
                "last_seen": inst.last_seen,
                "centroid": inst.centroid.tolist(),
                "bbox_min": inst.bbox_min.tolist(),
                "bbox_max": inst.bbox_max.tolist(),
                "label": inst.label,
                "score": inst.score,
                "is_active": inst.is_active,
                "keyframes": inst.keyframes,
            }
            for inst in self.valid_instances.values()
        ]

    def labeled_summaries(self) -> list[dict]:
        """Like summaries(), but restricted to instances with a semantic label."""
        return [s for s in self.summaries() if s["label"] is not None]

    def labeled_instance_ids(self) -> set[int]:
        """IDs of valid instances that carry a semantic label."""
        return {iid for iid, inst in self.valid_instances.items() if inst.label is not None}
