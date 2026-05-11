from __future__ import annotations

from dataclasses import dataclass

from .types import FrameRecord


@dataclass(frozen=True)
class KeyframeSelectionStats:
    selected_count: int
    rejected_count: int
    effective_hz: float


def select_keyframes(
    frames: list[FrameRecord],
    mode: str,
    target_hz: float,
) -> tuple[list[FrameRecord], KeyframeSelectionStats]:
    if not frames:
        return [], KeyframeSelectionStats(0, 0, 0.0)

    if mode == "all_frames":
        duration = max(frames[-1].timestamp - frames[0].timestamp, 1e-9)
        eff = len(frames) / duration if len(frames) > 1 else 0.0
        return frames, KeyframeSelectionStats(len(frames), 0, eff)

    if mode != "fixed_frequency":
        raise ValueError(f"Unknown keyframe mode '{mode}'")

    if target_hz <= 0:
        raise ValueError("target_hz must be > 0 for fixed_frequency mode")

    dt = 1.0 / target_hz
    start = frames[0].timestamp
    end = frames[-1].timestamp
    ticks: list[float] = []
    t = start
    while t <= end + 1e-9:
        ticks.append(t)
        t += dt

    selected_indices: set[int] = set()
    for tick in ticks:
        best_idx = min(range(len(frames)), key=lambda i: abs(frames[i].timestamp - tick))
        selected_indices.add(best_idx)

    selected = [frames[i] for i in sorted(selected_indices)]
    duration = max(selected[-1].timestamp - selected[0].timestamp, 1e-9) if len(selected) > 1 else 1e-9
    eff = len(selected) / duration if len(selected) > 1 else 0.0
    return selected, KeyframeSelectionStats(len(selected), len(frames) - len(selected), eff)
