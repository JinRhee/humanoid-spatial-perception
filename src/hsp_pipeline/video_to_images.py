from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

NSEC_PER_SEC = 1_000_000_000
TEMP_FRAME_GLOB = "frame_*.jpg"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an MP4 video into pipeline-ready images_<sec>_<nsec>.jpg frames."
    )
    parser.add_argument("--input_video", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--start_sec", default=0, type=int)
    parser.add_argument("--start_nsec", default=0, type=int)
    parser.add_argument("--jpeg_quality", default=2, type=int)
    return parser


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"Required tool '{name}' was not found on PATH. Please install ffmpeg/ffprobe first."
        )
    return path


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(cmd)}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _parse_frame_timestamp(raw: dict[str, object]) -> float | None:
    for key in ("best_effort_timestamp_time", "pkt_pts_time", "pts_time", "pkt_dts_time"):
        value = raw.get(key)
        if value in (None, "N/A"):
            continue
        return float(value)
    return None


def _probe_timestamps(ffprobe_bin: str, input_video: Path) -> list[float]:
    result = _run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time,pkt_pts_time,pts_time,pkt_dts_time",
            "-of",
            "json",
            str(input_video),
        ]
    )
    payload = json.loads(result.stdout or "{}")
    frames = payload.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"No video frames found in {input_video}")

    timestamps: list[float] = []
    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise RuntimeError(f"Unexpected ffprobe frame payload at index {idx}")
        ts = _parse_frame_timestamp(frame)
        if ts is None:
            raise RuntimeError(f"Missing timestamp for frame {idx} in {input_video}")
        timestamps.append(ts)

    for idx in range(1, len(timestamps)):
        if not timestamps[idx] > timestamps[idx - 1]:
            raise RuntimeError(
                "Video frame timestamps are not strictly increasing: "
                f"index {idx - 1} ({timestamps[idx - 1]}) -> index {idx} ({timestamps[idx]})"
            )
    return timestamps


def _extract_frames(ffmpeg_bin: str, input_video: Path, temp_dir: Path, jpeg_quality: int) -> list[Path]:
    _run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_video),
            "-vsync",
            "0",
            "-q:v",
            str(jpeg_quality),
            str(temp_dir / "frame_%08d.jpg"),
        ]
    )
    frames = sorted(temp_dir.glob(TEMP_FRAME_GLOB))
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames for {input_video}")
    return frames


def _to_sec_nsec(timestamp_seconds: float, start_sec: int, start_nsec: int) -> tuple[int, int]:
    total_nsec = start_sec * NSEC_PER_SEC + start_nsec + int(round(timestamp_seconds * NSEC_PER_SEC))
    sec, nsec = divmod(total_nsec, NSEC_PER_SEC)
    return int(sec), int(nsec)


def convert_video_to_images(
    input_video: Path,
    output_dir: Path,
    start_sec: int = 0,
    start_nsec: int = 0,
    jpeg_quality: int = 2,
) -> None:
    if input_video.suffix.lower() != ".mp4":
        raise ValueError(f"Expected an .mp4 input, got: {input_video}")
    if not input_video.exists() or not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if start_sec < 0:
        raise ValueError("start_sec must be >= 0")
    if not 0 <= start_nsec < NSEC_PER_SEC:
        raise ValueError("start_nsec must be in [0, 1_000_000_000)")
    if not 2 <= jpeg_quality <= 31:
        raise ValueError("jpeg_quality must be between 2 and 31 for ffmpeg -q:v")

    ffmpeg_bin = _require_tool("ffmpeg")
    ffprobe_bin = _require_tool("ffprobe")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps = _probe_timestamps(ffprobe_bin, input_video)

    with tempfile.TemporaryDirectory(prefix="video_to_images_") as temp_root:
        temp_dir = Path(temp_root)
        extracted = _extract_frames(ffmpeg_bin, input_video, temp_dir, jpeg_quality)

        if len(extracted) != len(timestamps):
            raise RuntimeError(
                f"Frame count mismatch: ffprobe returned {len(timestamps)} timestamps but ffmpeg extracted {len(extracted)} frames"
            )

        target_names: list[Path] = []
        prev_pair: tuple[int, int] | None = None
        first_pair: tuple[int, int] | None = None
        for ts in timestamps:
            pair = _to_sec_nsec(ts, start_sec, start_nsec)
            if prev_pair is not None and pair <= prev_pair:
                raise RuntimeError(
                    "Generated output timestamps are not strictly increasing: "
                    f"{prev_pair} -> {pair}"
                )
            target = output_dir / f"images_{pair[0]}_{pair[1]}.jpg"
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite existing frame: {target}")
            target_names.append(target)
            if first_pair is None:
                first_pair = pair
            prev_pair = pair

        for src, dst in zip(extracted, target_names):
            shutil.move(str(src), str(dst))

        print(
            f"Wrote {len(target_names)} frames from {input_video} to {output_dir} "
            f"starting at images_{first_pair[0]}_{first_pair[1]}.jpg"
        )


def main() -> None:
    args = build_parser().parse_args()
    try:
        convert_video_to_images(
            input_video=args.input_video.resolve(),
            output_dir=args.output_dir.resolve(),
            start_sec=args.start_sec,
            start_nsec=args.start_nsec,
            jpeg_quality=args.jpeg_quality,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
