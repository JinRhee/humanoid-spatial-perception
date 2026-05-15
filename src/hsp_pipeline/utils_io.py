from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_tum_trajectory(path: Path, trajectory: list[tuple[float, np.ndarray, np.ndarray]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ts, t_xyz, q_xyzw in trajectory:
            f.write(
                f"{ts:.9f} {t_xyz[0]:.9f} {t_xyz[1]:.9f} {t_xyz[2]:.9f} "
                f"{q_xyzw[0]:.9f} {q_xyzw[1]:.9f} {q_xyzw[2]:.9f} {q_xyzw[3]:.9f}\n"
            )


def write_pcd_xyz(path: Path, points: np.ndarray) -> None:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("PCD writer expects points shape (N,3)")

    with path.open("w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {pts.shape[0]}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {pts.shape[0]}\n")
        f.write("DATA ascii\n")
        for p in pts:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def write_ply_xyzrgb(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    pts = np.asarray(points, dtype=np.float32)
    rgb = np.asarray(colors, dtype=np.uint8)
    if pts.shape[0] != rgb.shape[0]:
        raise ValueError("points and colors must have same length")

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        f.writelines(
            f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n"
            for p, c in zip(pts, rgb)
        )
