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

from .backbone import run_backbone
from .config import PipelineConfig, dump_yaml_file, load_config
from .geometry import run_geometry, sanity_check_geometry
from .ingest import load_frames
from .io_utils import write_jsonl, write_pcd_xyz, write_ply_xyzrgb, write_tum_trajectory
from .keyframes import select_keyframes
from .matching import cosine_affinity, mutual_matches_with_dustbin, sinkhorn_logspace
from .tracking import InstanceTracker, MergeWeights


@dataclass
class StageTiming:
    name: str
    seconds: float


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



def run_pipeline(images_dir: Path, config_path: Path, output_dir: Path, preset: str | None) -> None:
    return

# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    # Parse arguments: dataset, config, calib, no-viz,
    # and segmentation hyperparams (conf, iou, imgsz)

    # Load SLAM config from yaml

    # ---------------------------------------------------------------- #
    # Setup                                                             #
    # ---------------------------------------------------------------- #

    # Create multiprocessing manager and queues (main<->viz)

    # Load dataset and subsample per config

    # Apply calibration if provided

    # Create SharedKeyframes and SharedStates

    # Build UnifiedMASt3RInfer:
    #   - load AsymmetricMASt3R weights into model.encoder
    #   - call model.prepare(device)
    #   - call model.share_memory() so backend process can access encoder

    # Pass model.encoder (not model) to FrameTracker, so the tracker's
    # internal mast3r_match_symmetric uses the same weights

    # Instantiate FastSAM segmentation pipeline (main process only)

    # Instantiate SegmentationStore (main process only)

    # Set up calibration matrix K if use_calib

    # Clean up any previous trajectory / reconstruction files

    # ---------------------------------------------------------------- #
    # Spawn processes                                                   #
    # ---------------------------------------------------------------- #

    # Spawn viz process if not --no-viz

    # Spawn backend process:
    #   - pass model.encoder so FactorGraph and retrieval database
    #     use the same weights without re-loading

    # ---------------------------------------------------------------- #
    # Seed frame 0 before the loop                                      #
    # ---------------------------------------------------------------- #

    # Load frame 0 image from dataset
    # Run FastSAM on frame 0 -> prev_masks_raw
    # Store raw image as prev_img_np
    # prev_frame_obj = None (SLAM frame not created until INIT fires)

    # ---------------------------------------------------------------- #
    # Main tracking loop                                                #
    # ---------------------------------------------------------------- #

    i = 0
    while True:

        # Check viz messages; handle pause / terminate signals

        # Break if all frames consumed

        # Load current frame (timestamp, image) from dataset

        # Run FastSAM on current image -> curr_masks_raw
        # (always run, even in RELOC, so masks are ready if mode changes)

        # Create SLAM frame object (pose, image, true_shape, feat=None, pos=None)

        # ------------------------------------------------------------ #
        # INIT                                                          #
        # ------------------------------------------------------------ #
        if mode == Mode.INIT:
            # Run mono inference to bootstrap pointmap (no pair yet)
            # Append frame to keyframes
            # Queue global optimisation for frame 0
            # Set mode -> TRACKING
            # Cache frame as prev_frame_obj
            # Cache curr_masks_raw as prev_masks_raw
            # Advance i; continue

            pass

        # ------------------------------------------------------------ #
        # TRACKING                                                      #
        # ------------------------------------------------------------ #
        elif mode == Mode.TRACKING:
            # Resize prev_masks_raw and curr_masks_raw to MASt3R feature
            # map resolution; add batch dimension

            # If both mask sets are non-empty:
            #   call model.infer_unified(prev_frame_obj, frame,
            #                            masks_i, masks_j)
            #   -> (X, C, D, Q), match_result
            #   store result in SegmentationStore keyed by (i-1, i)
            #   log match count
            # Else (one side has no masks):
            #   call model.infer_slam(prev_frame_obj, frame)
            #   -> X, C, D, Q  (SLAM branch only, no seg result stored)
            #   log skip reason

            # Call tracker.track(frame)
            # Note: tracker internally re-decodes (accepted redundancy for now;
            # future work: pass pre-computed X,C,D,Q directly to skip re-decode)
            # -> add_new_kf, match_info, try_reloc

            # If try_reloc: set mode -> RELOC

            # Update shared state with current frame

            pass

        # ------------------------------------------------------------ #
        # RELOC                                                         #
        # ------------------------------------------------------------ #
        elif mode == Mode.RELOC:
            # Run mono inference to update frame pointmap
            # (segmentation not used during relocalisation)
            # Push frame to shared state and queue reloc
            # If single_thread: spin until backend finishes reloc

            pass

        # ------------------------------------------------------------ #
        # Post-frame bookkeeping                                        #
        # ------------------------------------------------------------ #

        # If add_new_kf:
        #   append frame to keyframes
        #   queue global optimisation for new keyframe index
        #   if single_thread: spin until backend finishes

        # Optionally query SegmentationStore for this keyframe pair
        # and attach semantic labels to the keyframe for downstream use

        # Roll forward:
        #   prev_frame_obj = frame   (encoder cache carries over)
        #   prev_masks_raw = curr_masks_raw

        # Log FPS every 30 frames

        i += 1

    # ---------------------------------------------------------------- #
    # Shutdown                                                          #
    # ---------------------------------------------------------------- #

    # Save trajectory, reconstruction, and keyframe images if requested

    # Save raw frames to disk if save_frames

    # Join backend and viz processes

    # Print done