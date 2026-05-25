#!/usr/bin/env python3
"""
flyby_render.py — circular flyby renderer for point clouds and meshes.

Loads one or more .ply / .obj / .las files, renders a smooth circular orbit
around the scene, and compiles the frames into a GIF or MP4.

When multiple files are given, all videos share the same camera path (computed
from the combined bounding box) so they can be compared side-by-side.

Usage
-----
  # Single file → fire_flyby.gif
  python tools/flyby_render.py test/fire.ply

  # Paired files on the same orbit
  python tools/flyby_render.py test/fire.ply test/fire_instances.ply

  # Apply a rigid transform (16 row-major values, or a .npy/.txt file)
  python tools/flyby_render.py test/fire.ply --transform "0,-1,0,0, 0,0,-1,0, 1,0,0,0, 0,0,0,1"
  python tools/flyby_render.py test/fire.ply --transform transform.npy

  # Full options
  python tools/flyby_render.py test/fire.ply test/fire_instances.ply \\
      --frames 180 --fps 15 --width 1280 --height 720 \\
      --elevation 30 --fov 60 --format gif --outdir output/
"""

import argparse
import copy
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_geometry(path: str) -> o3d.geometry.Geometry3D:
    p = Path(path)
    ext = p.suffix.lower()

    if ext in (".ply", ".obj"):
        mesh = o3d.io.read_triangle_mesh(str(p))
        if len(mesh.triangles) > 0:
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            return mesh
        # No triangles — treat as point cloud (common for .ply from depth fusion)
        pcd = o3d.io.read_point_cloud(str(p))
        if len(pcd.points) > 0:
            return pcd
        sys.exit(f"Could not load geometry from {path}")

    if ext == ".las":
        try:
            import laspy
        except ImportError:
            sys.exit("Install laspy to read .las files:  pip install laspy")
        las = laspy.read(str(p))
        pts = np.vstack([las.x, las.y, las.z]).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if hasattr(las, "red"):
            rgb = np.vstack([las.red, las.green, las.blue]).T.astype(np.float64)
            scale = rgb.max() if rgb.max() > 1.0 else 1.0
            pcd.colors = o3d.utility.Vector3dVector(rgb / scale)
        return pcd

    sys.exit(f"Unsupported file extension '{ext}' — use .ply, .obj, or .las")


def parse_transform(spec: str) -> np.ndarray:
    """Parse a 4×4 rigid transform from 16 floats or a .npy/.txt file path."""
    p = Path(spec)
    if p.exists():
        T = np.load(str(p)) if p.suffix == ".npy" else np.loadtxt(str(p))
    else:
        vals = [float(v) for v in spec.replace(",", " ").split()]
        if len(vals) != 16:
            sys.exit(f"--transform needs 16 values for a 4×4 matrix, got {len(vals)}")
        T = np.array(vals).reshape(4, 4)
    if T.shape != (4, 4):
        sys.exit(f"--transform matrix must be 4×4, got shape {T.shape}")
    return T


# ---------------------------------------------------------------------------
# Camera path
# ---------------------------------------------------------------------------

def compute_orbit(center: np.ndarray, diag: float, n_frames: int,
                  elevation_deg: float = 25.0, radius_scale: float = 1.5):
    """Return (eye, lookat, up) tuples for a full 360° orbit around center."""
    radius = diag * radius_scale / 2.0
    elev_rad = math.radians(elevation_deg)
    z_offset = radius * math.tan(elev_rad)

    poses = []
    for i in range(n_frames):
        theta = 2 * math.pi * i / n_frames
        eye = np.array([
            center[0] + radius * math.cos(theta),
            center[1] + radius * math.sin(theta),
            center[2] + z_offset,
        ])
        lookat = center.copy()
        up = np.array([0.0, 0.0, 1.0])

        fwd = lookat - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        up /= np.linalg.norm(up)

        poses.append((eye, lookat, up))
    return poses


def combined_orbit(geos, n_frames, elevation_deg, radius_scale=1.5):
    """Compute one orbit that encompasses all loaded geometries."""
    corners = []
    for g in geos:
        bb = g.get_axis_aligned_bounding_box()
        corners.append(np.asarray(bb.min_bound))
        corners.append(np.asarray(bb.max_bound))
    corners = np.array(corners)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    center = (lo + hi) / 2.0
    diag = float(np.linalg.norm(hi - lo))
    return center, diag, compute_orbit(center, diag, n_frames, elevation_deg, radius_scale)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _material(geo: o3d.geometry.Geometry3D, point_size: float = 2.0):
    mat = o3d.visualization.rendering.MaterialRecord()
    if isinstance(geo, o3d.geometry.TriangleMesh):
        mat.shader = "defaultLit"
    else:
        mat.shader = "defaultUnlit"
        mat.point_size = point_size
    return mat


def render_frames(geo: o3d.geometry.Geometry3D, poses, out_dir: Path,
                  width: int, height: int, fov: float = 60.0):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.set_lighting(
        o3d.visualization.rendering.Open3DScene.LightingProfile.MED_SHADOWS,
        np.array([0.577, -0.577, -0.577], dtype=np.float32),
    )
    renderer.scene.add_geometry("scene", geo, _material(geo))

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(poses)
    for i, (eye, lookat, up) in enumerate(poses):
        renderer.setup_camera(
            fov,
            lookat.astype(np.float32),
            eye.astype(np.float32),
            up.astype(np.float32),
        )
        img = renderer.render_to_image()
        o3d.io.write_image(str(out_dir / f"frame_{i:05d}.png"), img)
        if (i + 1) % max(1, n // 10) == 0 or i == n - 1:
            print(f"    {i+1:4d}/{n} frames", end="\r", flush=True)
    print()


# ---------------------------------------------------------------------------
# Video / GIF compilation
# ---------------------------------------------------------------------------

def compile_output(frames_dir: Path, output_path: Path, fps: int):
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH — install it and retry")

    input_pattern = str(frames_dir / "frame_%05d.png")

    if output_path.suffix.lower() == ".gif":
        # Two-pass palette GIF for good colour quality
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-vf", (
                "split[s0][s1];"
                "[s0]palettegen=max_colors=256:stats_mode=diff[p];"
                "[s1][p]paletteuse=dither=bayer:bayer_scale=5"
            ),
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "slow",
            str(output_path),
        ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Circular flyby renderer — one GIF/MP4 per input file, all on the same orbit."
    )
    ap.add_argument("files", nargs="+", help="Input .ply / .obj / .las file(s)")
    ap.add_argument("--frames",    type=int,   default=120,   help="Total frames for one full orbit (default 120)")
    ap.add_argument("--fps",       type=int,   default=15,    help="Output FPS (default 15)")
    ap.add_argument("--width",     type=int,   default=None,  help="Frame width in pixels (default 640 for gif, 1280 for mp4)")
    ap.add_argument("--height",    type=int,   default=None,  help="Frame height in pixels (default 360 for gif, 720 for mp4)")
    ap.add_argument("--elevation", type=float, default=25.0,  help="Camera elevation above horizontal in degrees (default 25)")
    ap.add_argument("--fov",       type=float, default=60.0,  help="Vertical field of view in degrees (default 60)")
    ap.add_argument("--radius",    type=float, default=1.5,   help="Orbit radius as a multiple of scene half-diagonal (default 1.5)")
    ap.add_argument("--format",    choices=["gif", "mp4"], default="gif",
                                               help="Output format (default gif)")
    ap.add_argument("--outdir",    default="media",            help="Directory for output files (default: media/)")
    ap.add_argument(
        "--transform",
        default=None,
        metavar="MATRIX",
        help=(
            "Rigid transform applied to every geometry before rendering. "
            "Provide 16 row-major floats (comma- or space-separated) for a 4×4 "
            "homogeneous matrix, or a path to a .npy / .txt file. "
            "Example (90° rotation about Z): "
            "'0,-1,0,0, 1,0,0,0, 0,0,1,0, 0,0,0,1'"
        ),
    )
    args = ap.parse_args()

    # Format-dependent resolution defaults
    if args.width is None:
        args.width = 640 if args.format == "gif" else 1280
    if args.height is None:
        args.height = 360 if args.format == "gif" else 720

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- optional rigid transform -------------------------------------------
    T = None
    if args.transform is not None:
        T = parse_transform(args.transform)
        print(f"Transform:\n{np.round(T, 4)}")

    # ---- load (and optionally transform) ------------------------------------
    geos = []
    for fpath in args.files:
        print(f"Loading  {fpath} …")
        g = load_geometry(fpath)
        n = len(g.vertices) if isinstance(g, o3d.geometry.TriangleMesh) else len(g.points)
        print(f"  {type(g).__name__}  —  {n:,} points/vertices")
        if T is not None:
            g = copy.deepcopy(g)
            g.transform(T)
        geos.append(g)

    # ---- shared orbit -------------------------------------------------------
    center, diag, poses = combined_orbit(geos, args.frames, args.elevation, args.radius)
    orbit_radius = diag * args.radius / 2.0
    print(f"\nOrbit  center={np.round(center, 2)}  radius={orbit_radius:.2f} m  "
          f"elevation={args.elevation}°  frames={args.frames}  fps={args.fps}")

    # ---- render + encode each file ------------------------------------------
    ext = args.format
    for geo, fpath in zip(geos, args.files):
        stem = Path(fpath).stem
        frames_dir = out_dir / f"_frames_{stem}"
        output_path = out_dir / f"{stem}_flyby.{ext}"

        print(f"\nRendering  {stem}  ({args.frames} frames @ {args.width}×{args.height}) …")
        render_frames(geo, poses, frames_dir, args.width, args.height, args.fov)

        print(f"Encoding   {output_path} …")
        compile_output(frames_dir, output_path, args.fps)
        shutil.rmtree(frames_dir)
        print(f"  -> {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
