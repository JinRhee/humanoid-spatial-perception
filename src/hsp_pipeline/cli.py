from __future__ import annotations
import argparse
from pathlib import Path

from .pipeline import run_pipeline

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Semantic 3D reconstruction pipeline")

    p.add_argument("--dataset",  required=True,  type=Path)
    p.add_argument("--config",      required=True,  type=Path)
    p.add_argument("--output_dir",  required=True,  type=Path)
    p.add_argument("--no-viz",   action="store_true")
    p.add_argument("--calib",    default="")

    return p


def main() -> None:
    args = build_parser().parse_args()
    
    run_pipeline(args)


if __name__ == "__main__":
    main()