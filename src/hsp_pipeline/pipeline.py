from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig, dump_yaml_file, load_config
from .geometry import run_geometry, sanity_check_geometry
from .ingest import load_frames
from .io_utils import ensure_dir, write_jsonl, write_pcd_xyz, write_ply_xyzrgb, write_tum_trajectory
from .keyframes import select_keyframes
from .matching import cosine_affinity, mutual_matches_with_dustbin, sinkhorn_logspace
from .semantics import load_prompts, load_synonyms, run_semantics, sanity_check_semantics
from .tracking import InstanceTracker, MergeWeights


@dataclass
class StageTiming:
    name: str
    seconds: float


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _git_submodule_hash(repo_root: Path, sub_path: str) -> str:
    cmd = ["git", "-C", str(repo_root), "rev-parse", f"HEAD:{sub_path}"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return out.stdout.strip() if out.returncode == 0 else ""


def _write_error_report(out_dir: Path, stage: str, err: Exception, last_frame: str | None) -> None:
    with (out_dir / "error_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"failure_stage: {stage}\n")
        f.write(f"error: {type(err).__name__}: {err}\n")
        f.write(f"last_successful_frame: {last_frame or 'N/A'}\n")


def _log_stage(name: str, begin: float, stats: dict[str, Any]) -> StageTiming:
    dt = time.time() - begin
    print(f"[stage] {name} | {dt:.2f}s | {json.dumps(stats, default=str)}")
    return StageTiming(name=name, seconds=dt)


def _frame_groups(segments) -> dict[float, list]:
    grouped = defaultdict(list)
    for seg in segments:
        grouped[seg.frame_ts].append(seg)
    return dict(grouped)


def _run_pairwise_matching(grouped: dict[float, list]) -> list[dict[str, Any]]:
    ts = sorted(grouped)
    rows: list[dict[str, Any]] = []
    for i in range(len(ts) - 1):
        a_ts, b_ts = ts[i], ts[i + 1]
        g1 = np.stack([s.descriptor for s in grouped[a_ts]], axis=0) if grouped[a_ts] else np.zeros((0, 24))
        g2 = np.stack([s.descriptor for s in grouped[b_ts]], axis=0) if grouped[b_ts] else np.zeros((0, 24))
        aff = cosine_affinity(g1, g2)
        assign = sinkhorn_logspace(aff, iters=50)
        matches = mutual_matches_with_dustbin(assign)
        rows.append(
            {
                "frame_a": a_ts,
                "frame_b": b_ts,
                "num_segments_a": int(g1.shape[0]),
                "num_segments_b": int(g2.shape[0]),
                "accepted_matches": [{"i": i0, "j": j0, "prob": p} for i0, j0, p in matches],
            }
        )
    return rows


def _instances_to_ply(instances, path: Path) -> None:
    if not instances:
        write_ply_xyzrgb(path, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))
        return
    all_pts = []
    all_rgb = []
    for inst in instances:
        color = np.array([
            (inst.instance_id * 53) % 255,
            (inst.instance_id * 97) % 255,
            (inst.instance_id * 193) % 255,
        ], dtype=np.uint8)
        pts = inst.points_3d
        all_pts.append(pts)
        all_rgb.append(np.tile(color[None, :], (pts.shape[0], 1)))
    write_ply_xyzrgb(path, np.concatenate(all_pts, axis=0), np.concatenate(all_rgb, axis=0))


def _sanity_check_instances(instances, global_map: np.ndarray) -> None:
    if not instances:
        raise RuntimeError("Sanity check failed: instances.jsonl is empty")
    if global_map.size == 0:
        return
    map_min = global_map.min(axis=0)
    map_max = global_map.max(axis=0)
    for inst in instances:
        c = inst.centroid
        if np.any(c < map_min) or np.any(c > map_max):
            raise RuntimeError(
                f"Sanity check failed: centroid for instance {inst.instance_id} lies outside global map bounds"
            )


def _memory_snapshot() -> dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "cuda_allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
                "cuda_reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
            }
    except Exception:
        pass
    return {"cuda": "unavailable"}


def _estimate_model_footprint_mb(cfg: PipelineConfig) -> dict[str, float]:
    fp16 = bool(cfg.memory.get("fp16", False))
    scale = 0.7 if fp16 else 1.0
    return {
        "masts3r_backbone": 5000 * scale,
        "sam3_image_model": 2800 * scale,
        "segmast3r_heads": 600 * scale,
    }


def _guard_vram(cfg: PipelineConfig) -> None:
    est = _estimate_model_footprint_mb(cfg)
    total = sum(est.values())
    try:
        import torch

        if torch.cuda.is_available():
            avail = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            if total > avail:
                raise RuntimeError(
                    "Estimated model footprint exceeds available VRAM: "
                    + ", ".join(f"{k}={v:.1f}MB" for k, v in est.items())
                    + f", total={total:.1f}MB, available={avail:.1f}MB"
                )
    except ImportError:
        pass


def run_pipeline(images_dir: Path, config_path: Path, output_dir: Path, preset: str | None) -> None:
    output_dir = output_dir.resolve()
    ensure_dir(output_dir)

    timings: list[StageTiming] = []
    last_successful_frame = None

    cfg = load_config(config_path, preset)
    _guard_vram(cfg)

    try:
        t0 = time.time()
        frames = load_frames(images_dir)
        timings.append(_log_stage("ingest", t0, {"frames": len(frames), **_memory_snapshot()}))

        t0 = time.time()
        selected, kstats = select_keyframes(
            frames,
            mode=str(cfg.keyframe["mode"]),
            target_hz=float(cfg.keyframe.get("target_hz", 5.0)),
        )
        timings.append(
            _log_stage(
                "keyframe_select",
                t0,
                {
                    "selected": kstats.selected_count,
                    "rejected": kstats.rejected_count,
                    "effective_hz": round(kstats.effective_hz, 4),
                    **_memory_snapshot(),
                },
            )
        )

        t0 = time.time()
        geom = run_geometry(selected, cfg.geometry)
        sanity_check_geometry(geom.trajectory, geom.pointclouds)
        write_tum_trajectory(output_dir / "trajectory_tum.txt", geom.trajectory)
        pcd_dir = output_dir / "pointclouds"
        ensure_dir(pcd_dir)
        for fr in selected:
            pc = geom.pointclouds.get((fr.sec, fr.nsec), np.zeros((0, 3), dtype=np.float32))
            write_pcd_xyz(pcd_dir / f"pointcloud_{fr.sec}_{fr.nsec}.pcd", pc)
            last_successful_frame = f"{fr.sec}_{fr.nsec}"
        write_pcd_xyz(output_dir / "map_final.pcd", geom.global_map)
        traj_by_ts = {ts: t_xyz for ts, t_xyz, _ in geom.trajectory}
        timings.append(_log_stage("geometry", t0, {"poses": len(geom.trajectory), **_memory_snapshot()}))

        t0 = time.time()
        prompt_path = Path(str(cfg.semantics["prompt_file"]))
        synonym_path = Path(str(cfg.semantics["synonym_map"]))
        prompts = load_prompts(prompt_path, set(x.lower() for x in cfg.semantics.get("structural_labels", [])))
        synonyms = load_synonyms(synonym_path)
        sem = run_semantics(
            selected,
            prompts,
            synonyms,
            conf_min=float(cfg.semantics.get("point_conf_min", 0.1)),
            detection_conf_threshold=float(cfg.semantics.get("detection_conf_threshold", 0.01)),
            trajectory_by_ts=traj_by_ts,
            max_image_resolution=int(cfg.memory.get("max_image_resolution", 0)),
            max_masks_per_frame=int(cfg.memory.get("max_masks_per_frame", 0)),
        )
        sanity_check_semantics(sem.object_segments, keyframe_count=len(selected))
        for fr in selected:
            rows = sem.per_frame_rows.get((fr.sec, fr.nsec), [])
            write_jsonl(output_dir / f"segments_{fr.sec}_{fr.nsec}.jsonl", rows)
        write_pcd_xyz(output_dir / "background_geometry.pcd", sem.structural_points)
        timings.append(
            _log_stage(
                "semantics",
                t0,
                {"object_segments": len(sem.object_segments), "structural_points": int(sem.structural_points.shape[0]), **_memory_snapshot()},
            )
        )

        t0 = time.time()
        grouped = _frame_groups(sem.object_segments)
        pairwise = _run_pairwise_matching(grouped)
        write_jsonl(output_dir / "pairwise_matches.jsonl", pairwise)

        mw = cfg.fusion.get("merge_weights", {})
        tracker = InstanceTracker(
            ema_decay=float(cfg.fusion.get("ema_decay", 0.9)),
            merge_threshold=float(cfg.fusion.get("merge_threshold", 0.55)),
            max_centroid_distance=float(cfg.fusion.get("max_centroid_distance", 3.0)),
            label_mismatch_penalty=float(cfg.fusion.get("label_mismatch_penalty", 0.5)),
            min_support_count=int(cfg.fusion.get("min_support_count", 2)),
            resweep_interval=int(cfg.fusion.get("resweep_interval", 25)),
            skip_resweep=bool(cfg.fusion.get("skip_resweep", False)),
            weights=MergeWeights(
                descriptor=float(mw.get("descriptor", 0.40)),
                centroid=float(mw.get("centroid", 0.25)),
                iou=float(mw.get("iou", 0.25)),
                size=float(mw.get("size", 0.10)),
            ),
        )

        ts_sorted = sorted(grouped)
        for idx, ts in enumerate(ts_sorted):
            tracker.update_frame(grouped[ts], frame_idx=idx)

        keep, low = tracker.final_split()
        _sanity_check_instances(keep, geom.global_map)

        write_jsonl(output_dir / "instances.jsonl", [inst.as_json() for inst in keep])
        write_jsonl(output_dir / "low_confidence_instances.jsonl", [inst.as_json() for inst in low])
        _instances_to_ply(keep, output_dir / "instances.ply")

        label_dist = Counter(inst.canonical_label for inst in keep)
        mean_support = float(np.mean([inst.support_count for inst in keep])) if keep else 0.0
        print(
            f"[stats] instances={len(keep)} mean_support={mean_support:.2f} labels={dict(label_dist)}"
        )

        timings.append(
            _log_stage(
                "fusion_tracking_export",
                t0,
                {
                    "instances": len(keep),
                    "low_conf_instances": len(low),
                    **_memory_snapshot(),
                },
            )
        )

        t0 = time.time()
        repo_root = Path(__file__).resolve().parents[2]
        submodules = {
            "external/MASt3R-SLAM": _git_submodule_hash(repo_root, "external/MASt3R-SLAM"),
            "external/segmast3r": _git_submodule_hash(repo_root, "external/segmast3r"),
            "external/sam3": _git_submodule_hash(repo_root, "external/sam3"),
        }
        checkpoints = cfg.paths.get("checkpoints", {})
        checkpoint_info = {
            name: {"path": p, "sha256": _sha256(Path(p)) if p else ""}
            for name, p in checkpoints.items()
        }

        prompt_file = Path(str(cfg.semantics["prompt_file"]))
        synonym_file = Path(str(cfg.semantics["synonym_map"]))

        run_meta = {
            "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "submodule_commits": submodules,
            "model_checkpoints": checkpoint_info,
            "prompt_file_contents": prompt_file.read_text(encoding="utf-8") if prompt_file.exists() else "",
            "synonym_map_contents": synonym_file.read_text(encoding="utf-8") if synonym_file.exists() else "",
            "resolved_config": cfg.raw,
            "effective_keyframe_hz": kstats.effective_hz,
            "total_frames_ingested": len(frames),
            "total_keyframes_selected": len(selected),
            "total_instances_final": len(keep),
            "total_instances_low_confidence": len(low),
            "stage_timings_seconds": [{"stage": t.name, "seconds": t.seconds} for t in timings],
        }
        dump_yaml_file(output_dir / "run_metadata.yaml", run_meta)
        timings.append(_log_stage("metadata", t0, {"path": str(output_dir / "run_metadata.yaml")}))

    except Exception as exc:
        _write_error_report(output_dir, stage=(timings[-1].name if timings else "startup"), err=exc, last_frame=last_successful_frame)
        raise
