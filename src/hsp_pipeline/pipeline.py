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

import argparse
import torch
import torch.multiprocessing as mp


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    # ---------------------------------------------------------------- #
    # Arguments                                                         #
    # ---------------------------------------------------------------- #

    # --dataset       path to image sequence
    # --config        path to SLAM yaml config
    # --calib         optional calibration yaml
    # --no-viz        disable visualisation process
    # --save-as       output name
    # --seg-conf      FastSAM confidence threshold  (default 0.3)
    # --seg-iou       FastSAM IoU threshold         (default 0.4)
    # --seg-imgsz     FastSAM input resolution      (default 768)

    # ---------------------------------------------------------------- #
    # SLAM setup                                                        #
    # ---------------------------------------------------------------- #

    # Load SLAM config from yaml

    # Create mp.Manager and main<->viz queues

    # Load dataset; subsample per config

    # Apply calibration if --calib provided

    # Create SharedKeyframes and SharedStates (h, w from dataset)

    # ---------------------------------------------------------------- #
    # Model                                                             #
    # ---------------------------------------------------------------- #

    # Build UnifiedMASt3RInfer:
    #   load AsymmetricMASt3R weights into model.encoder
    #   call model.prepare(device)
    #   call model.share_memory()

    # Build FrameTracker with model.encoder
    # (tracker calls mast3r_match_asymmetric on model.encoder directly;
    #  passing model.encoder rather than model keeps that interface intact)

    # ---------------------------------------------------------------- #
    # FastSAM setup                                                     #
    # ---------------------------------------------------------------- #

    # Load FastSAM model from checkpoint path (vanilla ultralytics)
    # No wrapper class — call model(image, ...) directly in the loop

    # ---------------------------------------------------------------- #
    # Segmentation store and instance tracker                           #
    # ---------------------------------------------------------------- #

    # Instantiate SegmentationStore (keyed by frame index pair)

    # Instantiate InstanceTracker with merge weights, thresholds,
    # ema_decay, resweep_interval from config

    # ---------------------------------------------------------------- #
    # Calibration matrix                                                #
    # ---------------------------------------------------------------- #

    # If use_calib: build K tensor and pass to SharedKeyframes

    # ---------------------------------------------------------------- #
    # Clean up previous results                                         #
    # ---------------------------------------------------------------- #

    # Remove stale trajectory / reconstruction files if dataset.save_results

    # ---------------------------------------------------------------- #
    # Spawn processes                                                   #
    # ---------------------------------------------------------------- #

    # Spawn viz process if not --no-viz
    #   args: config, states, keyframes, main2viz, viz2main

    # Spawn backend process
    #   args: config, model.encoder, states, keyframes, K
    #   (backend uses model.encoder for FactorGraph and retrieval;
    #    it has no knowledge of segmentation)

    # ---------------------------------------------------------------- #
    # Seed frame 0                                                      #
    # ---------------------------------------------------------------- #

    # Load image 0 from dataset -> prev_img_np

    # Run FastSAM on prev_img_np:
    #   results = fastsam_model(prev_img_np, conf=..., iou=..., imgsz=...)
    #   prev_masks_raw = results[0].masks.data  (N, H, W) uint8 on device
    #   if no masks: prev_masks_raw = empty tensor (0, H, W)

    # prev_frame_obj = None  (SLAM frame created inside loop at i=0)

    # ---------------------------------------------------------------- #
    # Tracking loop                                                     #
    # ---------------------------------------------------------------- #

    i = 0
    while True:

        # Poll viz queue; update last_msg
        # Handle terminate signal -> break
        # Handle pause signal -> sleep and continue

        # Break if i == len(dataset)

        # Load (timestamp, img) from dataset[i]
        # curr_img_np = (img * 255).clip(0,255).astype(uint8)

        # Run FastSAM on curr_img_np:
        #   results = fastsam_model(curr_img_np, conf=..., iou=..., imgsz=...)
        #   curr_masks_raw = results[0].masks.data  (N, H, W) uint8 on device
        #   if no masks: curr_masks_raw = empty tensor (0, H, W)

        # Create SLAM frame object via create_frame(i, img, T_WC, ...)
        # T_WC = Sim3.Identity if i==0 else states.get_frame().T_WC

        # ------------------------------------------------------------ #
        # INIT (i == 0)                                                 #
        # ------------------------------------------------------------ #
        if mode == Mode.INIT:
            # mast3r_inference_mono -> X_init, C_init
            # frame.update_pointmap(X_init, C_init)
            # keyframes.append(frame)
            # states.queue_global_optimization(0)
            # states.set_mode(TRACKING)
            # states.set_frame(frame)

            # Cache:
            #   prev_frame_obj = frame
            #   prev_masks_raw = curr_masks_raw
            #   (prev_img_np already set before loop)

            # i += 1; continue
            pass

        # ------------------------------------------------------------ #
        # TRACKING                                                      #
        # ------------------------------------------------------------ #
        elif mode == Mode.TRACKING:

            # -- Unified inference --

            # Resize prev_masks_raw and curr_masks_raw to MASt3R
            # feature map resolution (H_feat, W_feat from prev_frame_obj.img)
            # Add batch dim -> (1, N, H_feat, W_feat)

            # If both mask tensors are non-empty:
            #   (X, C, D, Q), match_result, agg_desc_i, agg_desc_j =
            #       model.infer_unified(prev_frame_obj, frame,
            #                          masks_i, masks_j)
            #
            #   Note: infer_unified must be extended to also return
            #   agg_desc_i and agg_desc_j (pooled per-mask descriptors)
            #   so that SegmentRecord.descriptor can be filled without
            #   a second forward pass
            #
            #   Store in SegmentationStore keyed by (i-1, i)

            # Else (one side has no masks):
            #   X, C, D, Q = model.infer_slam(prev_frame_obj, frame)
            #   Skip seg store and instance tracker for this pair

            # -- Build SegmentRecords for current frame --

            # For each mask k in curr_masks_raw:
            #   pixel_indices = curr_masks_raw[k] > 0
            #   points_3d    = X[0][pixel_indices]  (res11 pointcloud,
            #                                        same direction as seg branch)
            #   mast3r_conf  = C[0][pixel_indices]
            #   descriptor   = agg_desc_j[0, :, k]  (24-dim pooled, from seg branch)
            #   centroid_xyz = points_3d.mean(0)
            #   bbox_min/max = points_3d.min/max(0)
            #   canonical_label = None  (FastSAM is class-agnostic;
            #                           label assignment is a future step
            #                           when SAM3 replaces FastSAM)
            #   sam3_score   = 1.0      (placeholder; FastSAM gives no
            #                           per-class score)
            #   Build SegmentRecord from above fields

            # -- Update instance tracker --
            # instance_tracker.update_frame(segment_records, i)

            # -- SLAM tracking (always runs) --
            # add_new_kf, match_info, try_reloc = tracker.track(frame)
            # Note: tracker re-runs its own asymmetric decode internally.
            # Encoder cache on frame means _encode_image is skipped.
            # Accepted redundancy for now; future: pass X,C,Q directly.

            # if try_reloc: states.set_mode(RELOC)
            # states.set_frame(frame)

            pass

        # ------------------------------------------------------------ #
        # RELOC                                                         #
        # ------------------------------------------------------------ #
        elif mode == Mode.RELOC:
            # mast3r_inference_mono -> X, C
            # frame.update_pointmap(X, C)
            # states.set_frame(frame)
            # states.queue_reloc()
            # if single_thread: spin until backend finishes reloc
            # (no segmentation during reloc)
            pass

        # ------------------------------------------------------------ #
        # Post-frame bookkeeping                                        #
        # ------------------------------------------------------------ #

        # If add_new_kf:
        #   keyframes.append(frame)
        #   states.queue_global_optimization(len(keyframes) - 1)
        #   if single_thread: spin until backend finishes

        # Roll forward (critical for encoder cache):
        #   prev_frame_obj = frame
        #   prev_masks_raw = curr_masks_raw

        # Log FPS every 30 frames

        i += 1

    # ---------------------------------------------------------------- #
    # Shutdown                                                          #
    # ---------------------------------------------------------------- #

    # instance_tracker.final_split() -> keep, low
    # Log or save global instances in keep

    # Save trajectory, reconstruction, keyframes if dataset.save_results

    # Save raw frames if save_frames

    # Join backend and viz processes

    # Print done