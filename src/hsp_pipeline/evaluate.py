import json
import pathlib
from typing import Optional

import cv2
import numpy as np
import torch

from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement

from .instance_tracker import InstanceTracker, instance_rgba


def prepare_savedir(args, dataset):
    save_dir = pathlib.Path("logs")
    if args.save_as != "default":
        save_dir = save_dir / args.save_as
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name = dataset.dataset_path.stem
    return save_dir, seq_name


def save_traj(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)
    save_ply(savedir / filename, pointclouds, colors)


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename = savedir / f"{t}.png"
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            ),
        )


def save_instances(savedir, filename_stem, instance_tracker: InstanceTracker):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)

    keep, low_support = instance_tracker.final_split()
    print(f"[save] {len(keep)} instances kept, {len(low_support)} discarded (low support)")

    # --- Manifest JSON ---
    manifest = []
    for inst in keep:
        manifest.append({
            "instance_id":   inst.instance_id,
            "label":         inst.label,
            "score":         float(inst.score),
            "support_count": inst.support_count,
            "last_seen":     inst.last_seen,
            "centroid":      inst.centroid.tolist(),
            "bbox_min":      inst.bbox_min.tolist(),
            "bbox_max":      inst.bbox_max.tolist(),
            "keyframes":     inst.keyframes,
            "num_points":    len(inst.points_3d),
        })
    manifest_path = savedir / f"{filename_stem}_instances.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[save] Manifest -> {manifest_path}")

    # --- Combined colored PLY ---
    all_points = []
    all_colors = []
    for inst in keep:
        if len(inst.points_3d) == 0:
            continue
        color_f = instance_rgba(inst.instance_id, alpha=1.0)[:3]
        color_u8 = (color_f * 255).clip(0, 255).astype(np.uint8)
        all_points.append(inst.points_3d)
        all_colors.append(np.tile(color_u8, (len(inst.points_3d), 1)))

    if all_points:
        combined_pts = np.concatenate(all_points, axis=0)
        combined_col = np.concatenate(all_colors, axis=0)
        save_ply(savedir / f"{filename_stem}_instances.ply", combined_pts, combined_col)
        print(f"[save] Combined instance cloud -> {savedir / f'{filename_stem}_instances.ply'}")
    else:
        print("[save] No instance points to save.")

    # --- Per-instance PLYs ---
    per_inst_dir = savedir / "instances"
    per_inst_dir.mkdir(exist_ok=True)
    saved = 0
    for inst in keep:
        if len(inst.points_3d) == 0:
            continue
        color_f = instance_rgba(inst.instance_id, alpha=1.0)[:3]
        color_u8 = (color_f * 255).clip(0, 255).astype(np.uint8)
        colors = np.tile(color_u8, (len(inst.points_3d), 1))
        save_ply(
            per_inst_dir / f"instance_{inst.instance_id:04d}.ply",
            inst.points_3d,
            colors,
        )
        saved += 1
    print(f"[save] {saved} per-instance PLYs -> {per_inst_dir}")


def save_ply(filename, points, colors):
    colors = colors.astype(np.uint8)
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)