from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from .types import FrameRecord

_IMAGE_RE = re.compile(r"^image_(\d+)_(\d+)\.jpg$")


def _parse_name(path: Path) -> tuple[int, int]:
    m = _IMAGE_RE.match(path.name)
    if not m:
        raise ValueError(
            f"Invalid filename '{path.name}'. Expected pattern images_<sec>_<nsec>.jpg"
        )
    return int(m.group(1)), int(m.group(2))


def _validate_readable(path: Path) -> None:
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as exc:
        raise ValueError(f"Unreadable/corrupt image '{path}': {exc}") from exc


def load_frames(images_dir: Path) -> list[FrameRecord]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    candidates = sorted(p for p in images_dir.glob("*.jpg") if p.is_file())
    if not candidates:
        raise ValueError(f"No .jpg images found in {images_dir}")

    parsed: list[FrameRecord] = []
    for p in candidates:
        sec, nsec = _parse_name(p)
        _validate_readable(p)
        parsed.append(FrameRecord(path=p, sec=sec, nsec=nsec, timestamp=sec + nsec * 1e-9))

    parsed.sort(key=lambda fr: (fr.sec, fr.nsec))

    bad_pairs: list[str] = []
    prev = parsed[0]
    for cur in parsed[1:]:
        if not (cur.timestamp > prev.timestamp):
            bad_pairs.append(f"{prev.path.name} -> {cur.path.name}")
        prev = cur

    if bad_pairs:
        raise ValueError(
            "Timestamps must be strictly monotonically increasing. Offending pairs: "
            + ", ".join(bad_pairs)
        )

    return parsed
