from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEMO_SCHEMA = "meha_human_demo_transition_v1"
EVENT_SCHEMA = "meha_human_demo_event_v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_demo_path(
    *,
    demo_dir: str | Path,
    prefix: str = "human_p61",
    seed: int = 0,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(demo_dir) / f"{prefix}_{stamp}_seed{int(seed)}.jsonl"


def frame_to_list(frame: Any) -> list[list[int]]:
    if isinstance(frame, list):
        return frame
    arr = np.asarray(frame, dtype=np.int16)
    return arr.astype(int, copy=False).tolist()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class JsonlDemoRecorder:
    """Append-only transition recorder for human play demonstrations."""

    def __init__(
        self,
        path: str | Path,
        *,
        active: bool = False,
        include_frames: bool = True,
        grid_size: int = 64,
        palette: str = "p61_arc16",
    ) -> None:
        self.path = Path(path)
        self.active = bool(active)
        self.include_frames = bool(include_frames)
        self.grid_size = int(grid_size)
        self.palette = str(palette)
        self._fh = None

    @property
    def is_open(self) -> bool:
        return self._fh is not None

    def ensure_open(self) -> None:
        if self._fh is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def set_active(self, active: bool, *, metadata: dict[str, Any] | None = None) -> None:
        active = bool(active)
        if active == self.active:
            return
        self.active = active
        self.write_event("recording_start" if active else "recording_stop", metadata or {}, force=True)
        if not active:
            self.close()

    def toggle(self, *, metadata: dict[str, Any] | None = None) -> bool:
        self.set_active(not self.active, metadata=metadata)
        return self.active

    def write_event(self, event_type: str, metadata: dict[str, Any] | None = None, *, force: bool = False) -> None:
        if not force and not self.active:
            return
        self.ensure_open()
        row = {
            "schema": EVENT_SCHEMA,
            "event_type": str(event_type),
            "timestamp_utc": utc_now_iso(),
            "metadata": json_safe(metadata or {}),
        }
        self._fh.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
        self._fh.flush()

    def write_transition(
        self,
        *,
        game_id: str,
        episode_id: str,
        step_idx: int,
        action_id: int,
        action_name: str,
        action_key: str,
        action_data: dict[str, Any] | None,
        frame_before: np.ndarray | None,
        frame_after: np.ndarray | None,
        extrinsic_reward: float,
        intrinsic_reward: float,
        train_reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        audio: dict[str, Any] | None,
        language: dict[str, Any] | None,
        affect: dict[str, Any] | None,
        timing: dict[str, Any] | None,
    ) -> None:
        if not self.active:
            return
        self.ensure_open()

        row = {
            "schema": DEMO_SCHEMA,
            "timestamp_utc": utc_now_iso(),
            "game_id": str(game_id),
            "episode_id": str(episode_id),
            "discrete_step": int(step_idx),
            "action_id": int(action_id),
            "action_name": str(action_name),
            "action_key": str(action_key),
            "action_data": json_safe(action_data or {}),
            "extrinsic_reward": float(extrinsic_reward),
            "intrinsic_reward": float(intrinsic_reward),
            "train_reward": float(train_reward),
            "terminated": int(terminated),
            "truncated": int(truncated),
            "info": json_safe(info or {}),
            "audio": json_safe(audio or {}),
            "language": json_safe(language or {}),
            "affect": json_safe(affect or {}),
            "timing": json_safe(timing or {}),
        }

        if self.include_frames:
            if frame_before is not None:
                row["observation_before"] = {
                    "schema": "meha_grid_v1",
                    "width": int(self.grid_size),
                    "height": int(self.grid_size),
                    "palette": self.palette,
                    "frame": frame_to_list(frame_before),
                }
            if frame_after is not None:
                row["observation_after"] = {
                    "schema": "meha_grid_v1",
                    "width": int(self.grid_size),
                    "height": int(self.grid_size),
                    "palette": self.palette,
                    "frame": frame_to_list(frame_after),
                }

        self._fh.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
