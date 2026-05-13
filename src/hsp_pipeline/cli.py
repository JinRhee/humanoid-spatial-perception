from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Semantic 3D reconstruction pipeline")
    p.add_argument("--images_dir", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--preset", default=None, choices=["quality", "balanced", "near_realtime"])
    return p


def main() -> None:
    args = build_parser().parse_args()
    from .pipeline import run_pipeline

    run_pipeline(
        images_dir=args.images_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        preset=args.preset,
    )


if __name__ == "__main__":
    main()
