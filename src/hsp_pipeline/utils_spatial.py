from __future__ import annotations

import numpy as np


def quaternion_xyzw_to_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(-1)
    if q.size != 4:
        raise ValueError("Quaternion must have 4 elements (x, y, z, w)")
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose_matrix(translation: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    rot = quaternion_xyzw_to_matrix(quaternion_xyzw)
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = t
    return mat


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.astype(np.float32)
    rot = pose[:3, :3]
    t = pose[:3, 3]
    out = (pts @ rot.T) + t[None, :]
    return out.astype(np.float32)
