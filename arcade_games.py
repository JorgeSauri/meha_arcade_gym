from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import numpy as np
P61_PALETTE_RGB = np.asarray(
    [
        [0, 0, 0],
        [0, 116, 217],
        [255, 65, 54],
        [46, 204, 64],
        [255, 220, 0],
        [170, 170, 170],
        [240, 18, 190],
        [255, 133, 27],
        [127, 219, 255],
        [135, 12, 37],
        [1, 255, 112],
        [255, 255, 255],
        [92, 52, 0],
        [191, 191, 191],
        [64, 224, 208],
        [128, 0, 128],
    ],
    dtype=np.float32,
) / 255.0
from arcade_gym.arcade_audio import AudioEvent, empty_audio_event, make_audio_event


MAX_REASONING_TOKENS = 1024


ACTION_LABELS = {
    1: "up",
    2: "down",
    3: "left",
    4: "right",
    5: "primary",
    6: "localized",
    7: "secondary",
    8: "wait",
}


ACTION_ALIAS_WORDS = [
    "amber",
    "brisk",
    "cobalt",
    "drift",
    "ember",
    "flux",
    "glint",
    "harbor",
    "ion",
    "jolt",
    "kappa",
    "lumen",
    "mosaic",
    "nova",
    "orbit",
    "pulse",
    "quartz",
    "ripple",
    "signal",
    "tilt",
    "umbra",
    "vector",
    "wisp",
    "xeno",
    "yonder",
    "zenith",
]


@dataclass
class ArcadeStepResult:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
    audio_event: AudioEvent = field(default_factory=empty_audio_event)


def clipped_train_reward(extrinsic: float, intrinsic: float, beta: float) -> float:
    return max(-1.0, min(1.0, float(extrinsic) + float(beta) * float(intrinsic)))


def _rects_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def _draw_rect(frame: np.ndarray, x: float, y: float, w: float, h: float, color: int) -> None:
    hgt, wid = frame.shape
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(wid, int(round(x + w)))
    y1 = min(hgt, int(round(y + h)))
    if x1 > x0 and y1 > y0:
        frame[y0:y1, x0:x1] = int(color)


def _draw_border(frame: np.ndarray, x: float, y: float, w: float, h: float, color: int, thickness: int = 2) -> None:
    _draw_rect(frame, x, y, w, thickness, color)
    _draw_rect(frame, x, y + h - thickness, w, thickness, color)
    _draw_rect(frame, x, y, thickness, h, color)
    _draw_rect(frame, x + w - thickness, y, thickness, h, color)


def _draw_filled_circle(frame: np.ndarray, cx: float, cy: float, r: float, color: int) -> None:
    hgt, wid = frame.shape
    ir = int(math.ceil(r))
    x0 = max(0, int(math.floor(cx - ir)))
    y0 = max(0, int(math.floor(cy - ir)))
    x1 = min(wid, int(math.ceil(cx + ir + 1)))
    y1 = min(hgt, int(math.ceil(cy + ir + 1)))
    rr = float(r) * float(r)
    for y in range(y0, y1):
        for x in range(x0, x1):
            dx = (x + 0.5) - float(cx)
            dy = (y + 0.5) - float(cy)
            if dx * dx + dy * dy <= rr:
                frame[y, x] = int(color)


def _background(size: tuple[int, int], color: int, *, accent: int | None = None, spacing: int = 16) -> np.ndarray:
    h, w = int(size[0]), int(size[1])
    frame = np.full((h, w), int(color), dtype=np.int16)
    if accent is None:
        return frame
    spacing = max(6, int(spacing))
    frame[::spacing, :] = int(accent)
    frame[:, :: spacing * 2] = int(accent)
    return frame


def _downsample(frame: np.ndarray, size: int = 128) -> list[list[int]]:
    arr = np.asarray(frame, dtype=np.int16)
    ys = np.linspace(0, arr.shape[0] - 1, int(size)).round().astype(int)
    xs = np.linspace(0, arr.shape[1] - 1, int(size)).round().astype(int)
    return arr[np.ix_(ys, xs)].astype(int, copy=False).tolist()


class ArcadeGame:
    game_id = "base"
    title = "Base Game"
    action_names = ACTION_LABELS

    def __init__(self, *, grid_size: int = 64, seed: int = 0) -> None:
        self.grid_size = int(grid_size)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.step_index = 0
        self.last_frame = np.zeros((self.grid_size, self.grid_size), dtype=np.int16)
        self.last_info: dict[str, Any] = {}
        self.action_aliases: dict[int, str] = dict(ACTION_LABELS)

    def _make_action_aliases(self) -> dict[int, str]:
        words = list(self.rng.choice(ACTION_ALIAS_WORDS, size=8, replace=False))
        return {i + 1: f"{words[i]}_{i + 1}" for i in range(8)}

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.step_index = 0
        self.last_info = {}
        self.action_aliases = self._make_action_aliases()
        self.last_frame = self._render_frame()
        return self._observation(), self.metadata()

    def step(self, action_id: int, action_data: dict[str, Any] | None = None) -> ArcadeStepResult:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "title": self.title,
            "seed": int(self.seed),
            "grid_size": int(self.grid_size),
            "action_names": {str(k): v for k, v in self.action_aliases.items()},
            "action_alias_seed": int(self.seed),
        }

    def action_name(self, action_id: int) -> str:
        return str(self.action_aliases.get(int(action_id), ACTION_LABELS.get(int(action_id), "unknown")))

    def action_semantic_name(self, action_id: int) -> str:
        return str(self.action_names.get(int(action_id), ACTION_LABELS.get(int(action_id), "unknown")))

    def accepts_busy_action(self, action_id: int) -> bool:
        return int(action_id) == 8

    def render_minimap(self, *, size: int = 128) -> list[list[int]]:
        return _downsample(self._render_frame(), size=size)

    def physics_busy(self) -> bool:
        return False

    def _render_frame(self) -> np.ndarray:
        return _background((self.grid_size, self.grid_size), 0)

    def _observation(self) -> dict[str, Any]:
        frame = self._render_frame().astype(np.int64, copy=False)
        self.last_frame = frame.astype(np.int16, copy=True)
        return {
            "frame": frame.tolist(),
            "grid": frame.tolist(),
            "pixels": frame.tolist(),
            "step": int(self.step_index),
            "available_actions": list(range(1, 9)),
            "game": self.metadata(),
        }

    def _finish_step(
        self,
        *,
        reward: float,
        intrinsic_reward: float,
        extrinsic_reward: float,
        terminated: bool,
        info: dict[str, Any],
        sound_kind: str = "silence",
        sound_intensity: float = 1.0,
    ) -> ArcadeStepResult:
        before = np.asarray(self.last_frame, dtype=np.int16).copy()
        obs = self._observation()
        after = np.asarray(obs["frame"], dtype=np.int16)
        changed_ratio = float(np.mean(before != after)) if before.shape == after.shape else 0.0
        audio = make_audio_event(sound_kind, intensity=sound_intensity) if sound_kind != "silence" else empty_audio_event()
        info = dict(info)
        info.update(
            {
                "game_id": self.game_id,
                "game_title": self.title,
                "extrinsic_reward": float(extrinsic_reward),
                "intrinsic_reward": float(intrinsic_reward),
                "changed_ratio": changed_ratio,
                "no_op": bool(changed_ratio <= 0.0001 and abs(float(reward)) <= 1e-8),
                "sound_event": audio.kind,
                "audio_features": audio.audio_features.astype(float).tolist(),
                "mel_spectrogram": audio.mel_spectrogram.astype(float).tolist(),
            }
        )
        self.last_info = info
        return ArcadeStepResult(
            observation=obs,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=False,
            info=info,
            audio_event=audio,
        )


class ColorSortGame(ArcadeGame):
    game_id = "p61_sort"
    title = "P6.1 Color Sort"
    action_names = {
        1: "move_up",
        2: "move_down",
        3: "move_left",
        4: "move_right",
        5: "push/interact",
        6: "poke_selected",
        7: "rotate_focus",
        8: "wait",
    }

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.step_index = 0
        self.action_aliases = self._make_action_aliases()
        s = self.grid_size / 128.0
        agent_size = max(5.0, round(5.0 * s * 1.5))
        self.agent = {
            "x": max(4.0, round(10.0 * s)),
            "y": float(self.grid_size // 2),
            "w": agent_size,
            "h": agent_size,
            "facing": 1,
        }
        colors = [3, 4, 8]
        self.containers: list[dict[str, Any]] = []
        slot_h = self.grid_size // len(colors)
        container_w = max(8, int(round(16 * s)))
        container_h_min = max(9, int(round(18 * s)))
        for i, color in enumerate(colors):
            self.containers.append(
                {
                    "x": self.grid_size - container_w - max(3, int(round(6 * s))),
                    "y": max(4, int(round(8 * s))) + i * slot_h,
                    "w": container_w,
                    "h": max(container_h_min, slot_h - max(7, int(round(14 * s)))),
                    "color": color,
                }
            )
        self.objects: list[dict[str, Any]] = []
        size_choices = [max(5, int(round(v * s * 1.5))) for v in (6, 8, 10, 12)]
        obj_x0 = max(10, int(round(24 * s)))
        obj_x1 = max(obj_x0 + 1, int(self.containers[0]["x"]) - max(size_choices) - 2)
        obj_y0 = max(6, int(round(12 * s)))
        obj_y1 = max(obj_y0 + 1, self.grid_size - max(size_choices) - max(4, int(round(8 * s))))
        for i, color in enumerate(colors):
            for _ in range(2):
                size = int(self.rng.choice(size_choices))
                w = float(size)
                h = float(size if self.rng.random() < 0.65 else max(5, size - max(1, int(round(3 * s)))))
                
                # Intentar encontrar una posición sin colisiones con otros objetos ni con el agente
                placed = False
                for _attempt in range(100):
                    x = float(self.rng.integers(obj_x0, obj_x1))
                    y = float(self.rng.integers(obj_y0, obj_y1))
                    rect = (x, y, w, h)
                    
                    # Verificar colisión con el agente
                    agent_rect = (self.agent["x"], self.agent["y"], self.agent["w"], self.agent["h"])
                    if _rects_overlap(rect, agent_rect):
                        continue
                        
                    # Verificar colisión con otros objetos ya creados
                    overlap = False
                    for existing in self.objects:
                        existing_rect = (existing["x"], existing["y"], existing["w"], existing["h"])
                        if _rects_overlap(rect, existing_rect):
                            overlap = True
                            break
                    if overlap:
                        continue
                        
                    # Posición válida encontrada
                    self.objects.append(
                        {
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h,
                            "color": color,
                            "delivered": False,
                        }
                    )
                    placed = True
                    break
                
                if not placed:
                    # Fallback si no se encontró posición libre en 100 intentos (muy improbable)
                    self.objects.append(
                        {
                            "x": float(self.rng.integers(obj_x0, obj_x1)),
                            "y": float(self.rng.integers(obj_y0, obj_y1)),
                            "w": w,
                            "h": h,
                            "color": color,
                            "delivered": False,
                        }
                    )
        self.last_info = {}
        self.last_frame = self._render_frame()
        return self._observation(), self.metadata()

    def _agent_rect(self) -> tuple[float, float, float, float]:
        return (self.agent["x"], self.agent["y"], self.agent["w"], self.agent["h"])

    def _wall_blocked(self, rect: tuple[float, float, float, float]) -> bool:
        x, y, w, h = rect
        return x < 2 or y < 2 or x + w > self.grid_size - 2 or y + h > self.grid_size - 2

    def _object_rect(self, obj: dict[str, Any]) -> tuple[float, float, float, float]:
        return (float(obj["x"]), float(obj["y"]), float(obj["w"]), float(obj["h"]))

    def _container_for_color(self, color: int) -> dict[str, Any]:
        for c in self.containers:
            if int(c["color"]) == int(color):
                return c
        return self.containers[0]

    def _delivered(self, obj: dict[str, Any]) -> bool:
        c = self._container_for_color(int(obj["color"]))
        cx = float(obj["x"]) + float(obj["w"]) * 0.5
        cy = float(obj["y"]) + float(obj["h"]) * 0.5
        return c["x"] <= cx <= c["x"] + c["w"] and c["y"] <= cy <= c["y"] + c["h"]

    def _can_place_object(self, target: dict[str, Any], nx: float, ny: float) -> bool:
        rect = (nx, ny, float(target["w"]), float(target["h"]))
        if self._wall_blocked(rect):
            return False
        for obj in self.objects:
            if obj is target or bool(obj.get("delivered", False)):
                continue
            if _rects_overlap(rect, self._object_rect(obj)):
                return False
        return True

    def _try_move(self, dx: float, dy: float) -> tuple[bool, bool]:
        nx = float(self.agent["x"]) + dx
        ny = float(self.agent["y"]) + dy
        rect = (nx, ny, float(self.agent["w"]), float(self.agent["h"]))
        if self._wall_blocked(rect):
            return False, False
        hit = None
        for obj in self.objects:
            if bool(obj.get("delivered", False)):
                continue
            if _rects_overlap(rect, self._object_rect(obj)):
                hit = obj
                break
        if hit is not None:
            ox = float(hit["x"]) + dx
            oy = float(hit["y"]) + dy
            if not self._can_place_object(hit, ox, oy):
                return False, False
            hit["x"], hit["y"] = ox, oy
            self.agent["x"], self.agent["y"] = nx, ny
            return True, True
        self.agent["x"], self.agent["y"] = nx, ny
        return True, False

    def step(self, action_id: int, action_data: dict[str, Any] | None = None) -> ArcadeStepResult:
        action_id = int(action_id)
        self.step_index += 1
        dx, dy = 0.0, 0.0
        speed = max(1.5, 3.0 * self.grid_size / 128.0)
        if action_id == 1:
            dy = -speed
            self.agent["facing"] = 0
        elif action_id == 2:
            dy = speed
            self.agent["facing"] = 2
        elif action_id == 3:
            dx = -speed
            self.agent["facing"] = 3
        elif action_id == 4:
            dx = speed
            self.agent["facing"] = 1
        elif action_id == 5:
            facing = int(self.agent.get("facing", 1))
            dx, dy = [(0.0, -speed), (speed, 0.0), (0.0, speed), (-speed, 0.0)][facing]
        elif action_id == 6:
            data = dict(action_data or {})
            tx = float(data.get("x", self.grid_size // 2))
            ty = float(data.get("y", self.grid_size // 2))
            ax = float(self.agent["x"]) + float(self.agent["w"]) * 0.5
            ay = float(self.agent["y"]) + float(self.agent["h"]) * 0.5
            if abs(tx - ax) >= abs(ty - ay):
                dx = speed if tx > ax else -speed
            else:
                dy = speed if ty > ay else -speed
        elif action_id == 7:
            self.agent["facing"] = (int(self.agent.get("facing", 1)) + 1) % 4
        moved, pushed = self._try_move(dx, dy) if (dx or dy) else (False, False)
        delivered_now = 0
        for obj in self.objects:
            if not bool(obj.get("delivered", False)) and self._delivered(obj):
                obj["delivered"] = True
                delivered_now += 1
        remaining = sum(1 for obj in self.objects if not bool(obj.get("delivered", False)))
        win = remaining == 0
        extrinsic = 1.0 if win else 0.0
        intrinsic = -0.01 + (0.03 if moved else -0.02) + (0.08 if pushed else 0.0) + 0.35 * delivered_now
        if not moved and action_id in (1, 2, 3, 4, 5, 6):
            intrinsic -= 0.05
        sound = "win" if win else ("bonus" if delivered_now else ("push" if pushed else ("wrong" if not moved and action_id != 8 else "step")))
        return self._finish_step(
            reward=extrinsic,
            intrinsic_reward=float(np.clip(intrinsic, -1.0, 1.0)),
            extrinsic_reward=extrinsic,
            terminated=win,
            sound_kind=sound,
            sound_intensity=1.0 if win or delivered_now else 0.55,
            info={
                "action_id": action_id,
                "action_name": self.action_name(action_id),
                "agent_moved": int(moved),
                "collision": int(not moved and action_id in (1, 2, 3, 4, 5, 6)),
                "object_moved": int(pushed),
                "object_rotated": 0,
                "object_delivered": int(delivered_now),
                "remaining_objects": int(remaining),
                "goal_reached": int(win),
                "agent_x": int(self.agent["x"]),
                "agent_y": int(self.agent["y"]),
                "target_x": int(self.grid_size - 12),
                "target_y": int(self.grid_size // 2),
            },
        )

    def _render_frame(self) -> np.ndarray:
        s = self.grid_size / 128.0
        frame = _background((self.grid_size, self.grid_size), 1, accent=12, spacing=max(8, int(round(16 * s))))
        _draw_border(frame, 0, 0, self.grid_size, self.grid_size, 13, thickness=2)
        for y in range(max(4, int(round(8 * s))), self.grid_size, max(8, int(round(16 * s)))):
            frame[y : y + 1, 2 : self.grid_size - 2] = 12
        for c in self.containers:
            _draw_border(frame, c["x"], c["y"], c["w"], c["h"], int(c["color"]), thickness=3)
        for obj in self.objects:
            color = int(obj["color"])
            if bool(obj.get("delivered", False)):
                color = 10
            _draw_rect(frame, obj["x"], obj["y"], obj["w"], obj["h"], color)
        _draw_rect(frame, self.agent["x"], self.agent["y"], self.agent["w"], self.agent["h"], 2)
        facing = int(self.agent.get("facing", 1))
        pointer = max(2, int(round(3 * s)))
        fx, fy = [(0, -pointer), (pointer, 0), (0, pointer), (-pointer, 0)][facing]
        _draw_rect(frame, self.agent["x"] + self.agent["w"] // 2 + fx, self.agent["y"] + self.agent["h"] // 2 + fy, max(1, pointer - 1), max(1, pointer - 1), 11)
        return frame

    def _observation(self) -> dict[str, Any]:
        obs = super()._observation()
        obs["agent"] = {"x": int(self.agent["x"]), "y": int(self.agent["y"]), "facing": int(self.agent.get("facing", 1))}
        obs["target"] = {"x": int(self.grid_size - 12), "y": int(self.grid_size // 2)}
        return obs


class NibblesGame(ArcadeGame):
    game_id = "nibbles"
    title = "Nibbles Color Diet"
    action_names = {
        1: "turn_up",
        2: "turn_down",
        3: "turn_left",
        4: "turn_right",
        5: "boost",
        6: "turn_to_selected",
        7: "slow",
        8: "wait",
    }

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.step_index = 0
        self.action_aliases = self._make_action_aliases()
        self.cell = 4
        self.cols = self.grid_size // self.cell
        self.rows = self.grid_size // self.cell
        self.snake = [(self.cols // 2 - i, self.rows // 2) for i in range(5)]
        self.direction = (1, 0)
        self.score = 0.0
        self.health = 3
        self.target_score = float(self.rng.choice([12, 15, 18]))
        self.foods: list[dict[str, Any]] = []
        for _ in range(9):
            self._spawn_food()
        self.move_clock = 0
        self.move_interval = 3
        self.last_frame = self._render_frame()
        return self._observation(), self.metadata()

    def _spawn_food(self) -> None:
        occupied = set(self.snake)
        occupied.update((int(f["x"]), int(f["y"])) for f in self.foods)
        for _ in range(200):
            x = int(self.rng.integers(2, self.cols - 2))
            y = int(self.rng.integers(2, self.rows - 2))
            if (x, y) in occupied:
                continue
            color = int(self.rng.choice([3, 1, 4, 2], p=[0.32, 0.22, 0.28, 0.18]))
            size = int(self.rng.choice([1, 1, 2, 3]))
            reward = {3: 1.0, 1: 1.5, 4: 2.0, 2: -2.0}[color] * float(size)
            self.foods.append({"x": x, "y": y, "size": size, "color": color, "value": reward})
            return

    def physics_busy(self) -> bool:
        return True

    def accepts_busy_action(self, action_id: int) -> bool:
        return True

    def step(self, action_id: int, action_data: dict[str, Any] | None = None) -> ArcadeStepResult:
        action_id = int(action_id)
        self.step_index += 1
        if action_id in (1, 2, 3, 4):
            nd = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[action_id]
            if (nd[0] + self.direction[0], nd[1] + self.direction[1]) != (0, 0):
                self.direction = nd
        elif action_id == 6:
            data = dict(action_data or {})
            hx, hy = self.snake[0]
            tx = int(float(data.get("x", self.grid_size // 2)) // self.cell)
            ty = int(float(data.get("y", self.grid_size // 2)) // self.cell)
            if abs(tx - hx) >= abs(ty - hy):
                self.direction = (1 if tx > hx else -1, 0)
            else:
                self.direction = (0, 1 if ty > hy else -1)

        # Progressive speed increase for Nibbles:
        self.move_interval = max(1, 3 - int(self.score // 4))
        self.move_clock += 1
        if self.move_clock < self.move_interval:
            # Skip movement this step, but still return a valid result
            return self._finish_step(
                reward=0.0,
                intrinsic_reward=0.0,
                extrinsic_reward=0.0,
                terminated=False,
                sound_kind="step",
                sound_intensity=0.2,
                info={
                    "action_id": action_id,
                    "action_name": self.action_name(action_id),
                    "score": float(self.score),
                    "health": int(self.health),
                    "goal_reached": 0,
                    "agent_x": int(self.snake[0][0] * self.cell),
                    "agent_y": int(self.snake[0][1] * self.cell),
                }
            )
        self.move_clock = 0

        boost = 2 if action_id == 5 else 1
        if action_id == 7:
            boost = 0
        collected = 0
        penalty = 0
        hit_wall = False
        for _ in range(boost):
            hx, hy = self.snake[0]
            nx, ny = hx + self.direction[0], hy + self.direction[1]
            if nx <= 0 or ny <= 0 or nx >= self.cols - 1 or ny >= self.rows - 1 or (nx, ny) in self.snake[:-1]:
                hit_wall = True
                self.health -= 1
                self.snake = [(self.cols // 2 - i, self.rows // 2) for i in range(4)]
                break
            self.snake.insert(0, (nx, ny))
            eaten = None
            for food in self.foods:
                if abs(nx - int(food["x"])) <= int(food["size"]) - 1 and abs(ny - int(food["y"])) <= int(food["size"]) - 1:
                    eaten = food
                    break
            if eaten is not None:
                self.foods.remove(eaten)
                self._spawn_food()
                value = float(eaten["value"])
                self.score += value
                if value >= 0:
                    collected += 1
                    extra = int(max(0.0, value) // 1.5)
                    for _ in range(extra):
                        self.snake.append(self.snake[-1])
                else:
                    penalty += 1
                    if len(self.snake) > 3:
                        self.snake.pop()
            else:
                self.snake.pop()
        win = self.score >= self.target_score
        extrinsic = 1.0 if win else 0.0
        intrinsic = -0.005 + 0.12 * collected - 0.18 * penalty - 0.25 * int(hit_wall) + 0.015 * boost
        sound = "win" if win else ("score" if collected else ("wrong" if penalty or hit_wall else "slide"))
        return self._finish_step(
            reward=extrinsic,
            intrinsic_reward=float(np.clip(intrinsic, -1.0, 1.0)),
            extrinsic_reward=extrinsic,
            terminated=win,
            sound_kind=sound,
            sound_intensity=1.0 if win or collected or penalty else 0.35,
            info={
                "action_id": action_id,
                "action_name": self.action_name(action_id),
                "score": float(self.score),
                "target_score": float(self.target_score),
                "health": int(self.health),
                "collected": int(collected),
                "penalty_food": int(penalty),
                "collision": int(hit_wall),
                "goal_reached": int(win),
                "agent_x": int(self.snake[0][0] * self.cell),
                "agent_y": int(self.snake[0][1] * self.cell),
                "target_x": int(self.grid_size - 8),
                "target_y": int(8),
            },
        )

    def _render_frame(self) -> np.ndarray:
        frame = _background((self.grid_size, self.grid_size), 12, accent=9, spacing=12)
        _draw_border(frame, 0, 0, self.grid_size, self.grid_size, 13, thickness=2)
        for food in self.foods:
            s = int(food["size"]) * self.cell
            _draw_rect(frame, int(food["x"]) * self.cell, int(food["y"]) * self.cell, s, s, int(food["color"]))
        for i, (x, y) in enumerate(self.snake):
            _draw_rect(frame, x * self.cell, y * self.cell, self.cell, self.cell, 11 if i == 0 else 10)
        bar_w = int(np.clip((self.score / max(1.0, self.target_score)) * (self.grid_size - 8), 0, self.grid_size - 8))
        _draw_rect(frame, 4, 4, bar_w, 2, 3)
        return frame

    def _observation(self) -> dict[str, Any]:
        obs = super()._observation()
        obs["agent"] = {"x": int(self.snake[0][0] * self.cell), "y": int(self.snake[0][1] * self.cell), "facing": 0}
        obs["target"] = {"x": int(self.grid_size - 8), "y": 8}
        return obs


class AngryBlocksGame(ArcadeGame):
    game_id = "angry_blocks"
    title = "Angry Blocks"
    action_names = {
        1: "aim_up",
        2: "aim_down",
        3: "power_down",
        4: "power_up",
        5: "launch",
        6: "aim_to_selected_launch",
        7: "reload",
        8: "wait_physics",
    }

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.step_index = 0
        self.action_aliases = self._make_action_aliases()
        self.scale = self.grid_size / 128.0
        s = self.scale
        self.ground_y = self.grid_size - max(8, int(round(16 * s)))
        self.origin = (max(8.0, round(18.0 * s)), float(self.ground_y - max(5, int(round(10 * s)))))
        self.aim_angle = -0.65
        self.power = max(4.0, 8.5 * s)
        self.shots_left = 6
        self.projectile: dict[str, Any] | None = None
        self.blocks: list[dict[str, Any]] = []
        x0 = int(self.rng.integers(max(34, int(round(78 * s))), max(35, int(round(92 * s)))))
        block_sizes = [max(4, int(round(v * s))) for v in (8, 10, 12, 14)]
        col_gap = max(6, int(round(13 * s)))
        row_gap = max(6, int(round(13 * s)))
        for row in range(4):
            for col in range(3):
                w = int(self.rng.choice(block_sizes))
                h = int(self.rng.choice(block_sizes))
                x = x0 + col * col_gap
                y = self.ground_y - (row + 1) * row_gap
                # Hacemos que haya múltiples bloques verdes (targets) en diferentes alturas
                # (por ejemplo, en la columna central y derecha en filas superiores)
                # y nos aseguramos de que algunos floten en el aire sin gravedad
                target = (row >= 2 and col == 1) or (row == 3 and col == 2) or (row == 1 and col == 0)
                is_floating = target and row >= 2 # Los bloques verdes superiores flotan en el aire
                self.blocks.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "w": float(w),
                        "h": float(h),
                        "vx": 0.0,
                        "vy": 0.0,
                        "color": 3 if target else int(self.rng.choice([4, 8, 14])),
                        "target": bool(target),
                        "floating": bool(is_floating),
                        "alive": True,
                    }
                )
        self.last_frame = self._render_frame()
        return self._observation(), self.metadata()

    def _launch(self) -> bool:
        if self.projectile is not None or self.shots_left <= 0:
            return False
        self.shots_left -= 1
        r = max(2.0, 3.0 * float(getattr(self, "scale", self.grid_size / 128.0)))
        self.projectile = {
            "x": self.origin[0],
            "y": self.origin[1],
            "vx": math.cos(self.aim_angle) * self.power,
            "vy": math.sin(self.aim_angle) * self.power,
            "r": r,
        }
        return True

    def _physics(self) -> tuple[int, int]:
        hits = 0
        target_hits = 0
        s = float(getattr(self, "scale", self.grid_size / 128.0))
        if self.projectile is not None:
            p = self.projectile
            for _ in range(3):
                p["vy"] = float(p["vy"]) + max(0.18, 0.35 * s)
                p["x"] = float(p["x"]) + float(p["vx"])
                p["y"] = float(p["y"]) + float(p["vy"])
                r = float(p.get("r", max(2.0, 3.0 * s)))
                prect = (float(p["x"]) - r, float(p["y"]) - r, r * 2, r * 2)
                for block in self.blocks:
                    if not bool(block.get("alive", True)):
                        continue
                    if _rects_overlap(prect, (block["x"], block["y"], block["w"], block["h"])):
                        block["alive"] = False
                        hits += 1
                        if bool(block.get("target", False)):
                            target_hits += 1
                        p["vx"] = -float(p["vx"]) * 0.35
                        p["vy"] = -abs(float(p["vy"])) * 0.25
                if p["x"] < -10 or p["x"] > self.grid_size + 10 or p["y"] > self.grid_size + 10:
                    self.projectile = None
                    break
        for block in self.blocks:
            if not bool(block.get("alive", True)):
                continue
            if bool(block.get("floating", False)):
                # Los bloques flotantes no caen por gravedad, se mantienen suspendidos en el aire
                block["vy"] = 0.0
                continue
            block["vy"] = min(max(3.0, 6.0 * s), float(block["vy"]) + max(0.09, 0.18 * s))
            floor_y = self.ground_y - float(block["h"])
            next_y = float(block["y"]) + float(block["vy"])
            if next_y >= floor_y:
                block["y"] = floor_y
                block["vy"] = 0.0
            else:
                block["y"] = next_y
        return hits, target_hits

    def physics_busy(self) -> bool:
        if self.projectile is not None:
            return True
        return any(
            bool(block.get("alive", True)) and abs(float(block.get("vy", 0.0))) > 0.05
            for block in self.blocks
        )

    def accepts_busy_action(self, action_id: int) -> bool:
        if self.projectile is not None:
            return int(action_id) == 8
        return int(action_id) in (1, 2, 3, 4, 5, 6, 7, 8)

    def step(self, action_id: int, action_data: dict[str, Any] | None = None) -> ArcadeStepResult:
        action_id = int(action_id)
        self.step_index += 1
        launched = False
        blocked = bool(self.physics_busy() and not self.accepts_busy_action(action_id))
        if blocked:
            action_id = 8
        if action_id == 1:
            self.aim_angle -= 0.10
        elif action_id == 2:
            self.aim_angle += 0.10
        elif action_id == 3:
            s = float(getattr(self, "scale", self.grid_size / 128.0))
            self.power = max(max(3.0, 4.0 * s), self.power - max(0.4, 0.8 * s))
        elif action_id == 4:
            s = float(getattr(self, "scale", self.grid_size / 128.0))
            self.power = min(max(6.0, 14.0 * s), self.power + max(0.4, 0.8 * s))
        elif action_id == 5:
            launched = self._launch()
        elif action_id == 6:
            data = dict(action_data or {})
            tx = float(data.get("x", self.grid_size - 20))
            ty = float(data.get("y", self.grid_size // 2))
            self.aim_angle = math.atan2(ty - self.origin[1], tx - self.origin[0])
            s = float(getattr(self, "scale", self.grid_size / 128.0))
            self.power = float(np.clip(math.hypot(tx - self.origin[0], ty - self.origin[1]) / max(3.75, 7.5 * s), max(3.0, 5.0 * s), max(6.0, 14.0 * s)))
            launched = self._launch()
        elif action_id == 7 and self.projectile is None:
            self.aim_angle = -0.65
            self.power = 8.5
        self.aim_angle = float(np.clip(self.aim_angle, -1.35, 0.15))
        hits, target_hits = self._physics()
        remaining_targets = sum(1 for b in self.blocks if bool(b.get("alive", True)) and bool(b.get("target", False)))
        win = remaining_targets == 0
        lost = self.shots_left <= 0 and self.projectile is None and not win
        terminated = win or lost
        extrinsic = 1.0 if win else 0.0
        intrinsic = -0.01 + 0.10 * hits + 0.25 * target_hits + (0.04 if launched else 0.0) - 0.50 * int(lost)
        sound = "win" if win else ("wrong" if lost else ("hit" if hits else ("launch" if launched else "slide")))
        s_info = float(getattr(self, "scale", self.grid_size / 128.0))
        target_x = int(self.grid_size - max(14, int(round(28 * s_info))))
        target_y = int(self.ground_y - max(20, int(round(48 * s_info))))
        return self._finish_step(
            reward=extrinsic,
            intrinsic_reward=float(np.clip(intrinsic, -1.0, 1.0)),
            extrinsic_reward=extrinsic,
            terminated=terminated,
            sound_kind=sound,
            sound_intensity=1.0 if win or lost or hits else 0.55,
            info={
                "action_id": action_id,
                "action_name": self.action_name(action_id),
                "action_blocked": int(blocked),
                "physics_busy": int(self.physics_busy()),
                "launch": int(launched),
                "hits": int(hits),
                "target_hits": int(target_hits),
                "remaining_targets": int(remaining_targets),
                "shots_left": int(self.shots_left),
                "collision": int(hits > 0),
                "goal_reached": int(win),
                "agent_x": int(self.origin[0]),
                "agent_y": int(self.origin[1]),
                "target_x": target_x,
                "target_y": target_y,
                "aim_angle": float(self.aim_angle),
                "power": float(self.power),
            },
        )

    def _render_frame(self) -> np.ndarray:
        s = float(getattr(self, "scale", self.grid_size / 128.0))
        frame = _background((self.grid_size, self.grid_size), 9, accent=12, spacing=max(9, int(round(18 * s))))
        _draw_rect(frame, 0, self.ground_y, self.grid_size, self.grid_size - self.ground_y, 12)
        launcher = max(3, int(round(6 * s)))
        _draw_rect(frame, self.origin[0] - launcher / 2, self.origin[1] - launcher / 2, launcher, launcher, 2)
        tx = self.origin[0] + math.cos(self.aim_angle) * self.power * max(4.0, 4.0 / max(0.5, s))
        ty = self.origin[1] + math.sin(self.aim_angle) * self.power * max(4.0, 4.0 / max(0.5, s))
        steps = 16
        for i in range(steps):
            a = i / max(1, steps - 1)
            x = self.origin[0] * (1 - a) + tx * a
            y = self.origin[1] * (1 - a) + ty * a
            _draw_rect(frame, x, y, 1, 1, 11)
        for block in self.blocks:
            if bool(block.get("alive", True)):
                _draw_rect(frame, block["x"], block["y"], block["w"], block["h"], int(block["color"]))
        if self.projectile is not None:
            p = self.projectile
            r = float(p.get("r", max(2.0, 3.0 * s)))
            _draw_rect(frame, float(p["x"]) - r, float(p["y"]) - r, r * 2, r * 2, 15)
        _draw_rect(frame, 2, 2, max(0, self.shots_left) * max(3, int(round(5 * s))), max(2, int(round(3 * s))), 4)
        return frame

    def _observation(self) -> dict[str, Any]:
        obs = super()._observation()
        obs["agent"] = {"x": int(self.origin[0]), "y": int(self.origin[1]), "facing": 1}
        s = float(getattr(self, "scale", self.grid_size / 128.0))
        obs["target"] = {"x": int(self.grid_size - max(14, int(round(28 * s)))), "y": int(self.ground_y - max(20, int(round(48 * s))))}
        return obs


class PlatformerGame(ArcadeGame):
    game_id = "platformer"
    title = "Mini Platform Blocks"
    action_names = {
        1: "jump",
        2: "drop",
        3: "run_left",
        4: "run_right",
        5: "jump_primary",
        6: "dash_to_selected",
        7: "brake",
        8: "wait",
    }

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.step_index = 0
        self.action_aliases = self._make_action_aliases()
        self.scale = self.grid_size / 128.0
        s = self.scale
        sc = lambda v: float(round(float(v) * s, 2))
        sc_size = lambda v: float(max(2, round(float(v) * s, 2)))
        self.world_w = max(self.grid_size * 2, int(round(256 * s)))
        self.world_h = self.grid_size
        self.player = {"x": sc(10), "y": sc(104), "w": max(4.0, sc_size(5)), "h": max(6.0, sc_size(8)), "vx": 0.0, "vy": 0.0, "grounded": True}
        self.platforms = [
            (sc(0), sc(112), sc(60), sc_size(8)),
            (sc(70), sc(100), sc(42), sc_size(7)),
            (sc(126), sc(88), sc(38), sc_size(7)),
            (sc(178), sc(102), sc(34), sc_size(7)),
            (sc(220), sc(92), sc(36), sc_size(8)),
        ]
        self.hazards = [
            (sc(58), sc(108), max(8.0, sc_size(10)), max(4.0, sc_size(4))),
            (sc(164), sc(84), max(8.0, sc_size(10)), max(4.0, sc_size(4))),
            (sc(212), sc(98), max(8.0, sc_size(10)), max(4.0, sc_size(4))),
        ]
        self.pickups = [
            {"x": sc(82), "y": sc(92), "w": max(4.0, sc_size(4)), "h": max(4.0, sc_size(4)), "color": 4, "value": 0.25, "collected": False},
            {"x": sc(140), "y": sc(80), "w": max(4.0, sc_size(5)), "h": max(4.0, sc_size(5)), "color": 3, "value": 0.35, "collected": False},
            {"x": sc(190), "y": sc(94), "w": max(4.0, sc_size(6)), "h": max(4.0, sc_size(6)), "color": 8, "value": 0.45, "collected": False},
        ]
        self.goal = (sc(236), sc(76), max(8.0, sc_size(10)), max(12.0, sc_size(16)))
        self.last_x = float(self.player["x"])
        self.last_frame = self._render_frame()
        return self._observation(), self.metadata()

    def _player_rect(self) -> tuple[float, float, float, float]:
        return (self.player["x"], self.player["y"], self.player["w"], self.player["h"])

    def _collide_platforms(self, rect: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
        for p in self.platforms:
            if _rects_overlap(rect, p):
                return p
        return None

    def physics_busy(self) -> bool:
        p = self.player
        return (
            not bool(p.get("grounded", False))
            or abs(float(p.get("vy", 0.0))) > 0.05
            or abs(float(p.get("vx", 0.0))) > 0.45
        )

    def accepts_busy_action(self, action_id: int) -> bool:
        return int(action_id) in (3, 4, 7, 8)

    def _physics(self) -> bool:
        p = self.player
        s = float(getattr(self, "scale", self.grid_size / 128.0))
        p["vy"] = min(max(3.0, 6.0 * s), float(p["vy"]) + max(0.22, 0.45 * s))
        p["x"] = float(np.clip(float(p["x"]) + float(p["vx"]), 0, self.world_w - float(p["w"])))
        rect = self._player_rect()
        hit = self._collide_platforms(rect)
        if hit is not None:
            hx, hy, hw, hh = hit
            if float(p["vx"]) > 0:
                p["x"] = hx - float(p["w"])
            elif float(p["vx"]) < 0:
                p["x"] = hx + hw
            p["vx"] = 0.0
        p["y"] = float(p["y"]) + float(p["vy"])
        p["grounded"] = False
        rect = self._player_rect()
        hit = self._collide_platforms(rect)
        if hit is not None:
            hx, hy, hw, hh = hit
            if float(p["vy"]) >= 0:
                p["y"] = hy - float(p["h"])
                p["grounded"] = True
            else:
                p["y"] = hy + hh
            p["vy"] = 0.0
        if float(p["y"]) > self.world_h + 10:
            p["x"], p["y"], p["vx"], p["vy"], p["grounded"] = 10.0 * s, 104.0 * s, 0.0, 0.0, True
            return True
        p["vx"] = float(p["vx"]) * 0.82
        return False

    def step(self, action_id: int, action_data: dict[str, Any] | None = None) -> ArcadeStepResult:
        action_id = int(action_id)
        self.step_index += 1
        p = self.player
        s = float(getattr(self, "scale", self.grid_size / 128.0))
        jumped = False
        blocked = bool(self.physics_busy() and not self.accepts_busy_action(action_id))
        if blocked:
            action_id = 8
        if action_id == 3:
            p["vx"] = max(-max(2.0, 4.0 * s), float(p["vx"]) - max(0.7, 1.4 * s))
        elif action_id == 4:
            p["vx"] = min(max(2.0, 4.0 * s), float(p["vx"]) + max(0.7, 1.4 * s))
        elif action_id in (1, 5) and bool(p.get("grounded", False)):
            p["vy"] = -max(3.75, 7.5 * s)
            p["grounded"] = False
            jumped = True
        elif action_id == 2:
            p["vy"] = min(max(3.0, 6.0 * s), float(p["vy"]) + max(0.75, 1.5 * s))
        elif action_id == 6:
            data = dict(action_data or {})
            tx = float(data.get("x", self.grid_size // 2))
            dash = max(2.25, 4.5 * s)
            p["vx"] = dash if tx > self.grid_size // 2 else -dash
        elif action_id == 7:
            p["vx"] = 0.0
        fell = self._physics()
        collected = 0
        prect = self._player_rect()
        for item in self.pickups:
            if not bool(item.get("collected", False)) and _rects_overlap(prect, (item["x"], item["y"], item["w"], item["h"])):
                item["collected"] = True
                collected += 1
        hurt = fell
        for hz in self.hazards:
            if _rects_overlap(prect, hz):
                p["x"], p["y"], p["vx"], p["vy"], p["grounded"] = 10.0 * s, 104.0 * s, 0.0, 0.0, True
                hurt = True
        win = _rects_overlap(self._player_rect(), self.goal)
        progress = max(-1.0, min(1.0, (float(p["x"]) - self.last_x) / max(20.0, 40.0 * s)))
        self.last_x = float(p["x"])
        extrinsic = 1.0 if win else 0.0
        intrinsic = -0.01 + 0.10 * collected + 0.04 * max(0.0, progress) + (0.04 if jumped else 0.0) - 0.25 * int(hurt)
        sound = "win" if win else ("score" if collected else ("wrong" if hurt else ("jump" if jumped else ("slide" if self.physics_busy() else "step"))))
        return self._finish_step(
            reward=extrinsic,
            intrinsic_reward=float(np.clip(intrinsic, -1.0, 1.0)),
            extrinsic_reward=extrinsic,
            terminated=win,
            sound_kind=sound,
            sound_intensity=1.0 if win or collected or hurt else 0.45,
            info={
                "action_id": action_id,
                "action_name": self.action_name(action_id),
                "action_blocked": int(blocked),
                "physics_busy": int(self.physics_busy()),
                "jumped": int(jumped),
                "collected": int(collected),
                "hazard": int(hurt),
                "collision": int(hurt),
                "goal_reached": int(win),
                "progress": float(progress),
                "agent_x": int(p["x"]),
                "agent_y": int(p["y"]),
                "target_x": int(self.goal[0]),
                "target_y": int(self.goal[1]),
            },
        )

    def _world_frame(self) -> np.ndarray:
        s = float(getattr(self, "scale", self.grid_size / 128.0))
        frame = _background((self.world_h, self.world_w), 1, accent=9, spacing=max(7, int(round(14 * s))))
        stripe = max(4, int(round(8 * s)))
        for x in range(0, self.world_w, max(8, int(round(16 * s)))):
            frame[0:1, x : x + stripe] = 12
        for plat in self.platforms:
            _draw_rect(frame, *plat, 13)
        for hz in self.hazards:
            _draw_rect(frame, *hz, 2)
        for item in self.pickups:
            if not bool(item.get("collected", False)):
                _draw_rect(frame, item["x"], item["y"], item["w"], item["h"], int(item["color"]))
        _draw_rect(frame, *self.goal, 10)
        _draw_rect(frame, self.player["x"], self.player["y"], self.player["w"], self.player["h"], 11)
        return frame

    def _render_frame(self) -> np.ndarray:
        world = self._world_frame()
        cam = int(np.clip(float(self.player["x"]) - self.grid_size * 0.38, 0, self.world_w - self.grid_size))
        return world[:, cam : cam + self.grid_size].copy()

    def render_minimap(self, *, size: int = 128) -> list[list[int]]:
        return _downsample(self._world_frame(), size=size)

    def _observation(self) -> dict[str, Any]:
        obs = super()._observation()
        obs["agent"] = {"x": int(self.player["x"]), "y": int(self.player["y"]), "facing": 1 if self.player["vx"] >= 0 else 3}
        obs["target"] = {"x": int(self.goal[0]), "y": int(self.goal[1])}
        return obs


class PhoenixDuelGame(ArcadeGame):
    game_id = "phoenix_duel"
    title = "Phoenix Sync Duel"
    action_names = {
        1: "rise",
        2: "drop",
        3: "strafe_left",
        4: "strafe_right",
        5: "fire",
        6: "aim_fire_selected",
        7: "shield",
        8: "wait",
    }

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.step_index = 0
        self.action_aliases = self._make_action_aliases()
        self.player = {"x": float(self.grid_size // 2 - 3), "y": float(self.grid_size - 9), "w": 6.0, "h": 4.0}
        self.cooldown = 0
        self.shield_ticks = 0
        self.lives = 3
        self.score = 0.0
        self.target_score = 10.0
        self.wave_dir = 1.0
        self.bullets: list[dict[str, Any]] = []
        self.enemy_bullets: list[dict[str, Any]] = []
        self.enemies: list[dict[str, Any]] = []
        for row in range(3):
            for col in range(6):
                self.enemies.append(
                    {
                        "x": float(8 + col * 9),
                        "y": float(8 + row * 7),
                        "w": 5.0,
                        "h": 4.0,
                        "phase": float(self.rng.random() * math.tau),
                        "color": int(self.rng.choice([3, 4, 8, 14])),
                        "alive": True,
                    }
                )
        self.last_frame = self._render_frame()
        return self._observation(), self.metadata()

    def _player_rect(self) -> tuple[float, float, float, float]:
        return (float(self.player["x"]), float(self.player["y"]), float(self.player["w"]), float(self.player["h"]))

    def _fire(self) -> bool:
        if self.cooldown > 0:
            return False
        self.cooldown = 3
        self.bullets.append(
            {
                "x": float(self.player["x"]) + float(self.player["w"]) * 0.5 - 1.0,
                "y": float(self.player["y"]) - 2.0,
                "w": 2.0,
                "h": 4.0,
                "vx": 0.0,
                "vy": -4.8,
            }
        )
        return True

    def physics_busy(self) -> bool:
        return True

    def accepts_busy_action(self, action_id: int) -> bool:
        return True

    def step(self, action_id: int, action_data: dict[str, Any] | None = None) -> ArcadeStepResult:
        action_id = int(action_id)
        self.step_index += 1
        fired = False
        shielded = False
        speed = 3.0
        if action_id == 1:
            self.player["y"] = max(self.grid_size * 0.55, float(self.player["y"]) - speed)
        elif action_id == 2:
            self.player["y"] = min(self.grid_size - float(self.player["h"]) - 2, float(self.player["y"]) + speed)
        elif action_id == 3:
            self.player["x"] = max(2.0, float(self.player["x"]) - speed)
        elif action_id == 4:
            self.player["x"] = min(self.grid_size - float(self.player["w"]) - 2, float(self.player["x"]) + speed)
        elif action_id == 5:
            fired = self._fire()
        elif action_id == 6:
            data = dict(action_data or {})
            tx = float(data.get("x", self.player["x"]))
            center = float(self.player["x"]) + float(self.player["w"]) * 0.5
            self.player["x"] = float(np.clip(float(self.player["x"]) + np.sign(tx - center) * speed, 2, self.grid_size - float(self.player["w"]) - 2))
            fired = self._fire()
        elif action_id == 7:
            self.shield_ticks = 4
            shielded = True

        self.cooldown = max(0, int(self.cooldown) - 1)
        self.shield_ticks = max(0, int(self.shield_ticks) - 1)

        # Progressive speed factor: starts at 0.45 and increases to 1.0 as score increases
        speed_factor = 0.45 + 0.55 * min(1.0, max(0.0, self.score) / max(1.0, self.target_score))

        alive = [e for e in self.enemies if bool(e.get("alive", True))]
        if alive:
            xs = [float(e["x"]) for e in alive]
            ys = [float(e["y"]) for e in alive]
            
            # Si alguna de las naves enemigas llega abajo (v.g., y > grid_size * 0.75),
            # las volvemos a subir a la parte superior (y = 8.0) para que no termine el juego
            # y sigan descendiendo en oleadas continuas.
            if max(ys) > self.grid_size * 0.75:
                for e in alive:
                    # Re-posicionamos los enemigos vivos arriba de forma escalonada
                    e["y"] = 8.0 + (float(e["y"]) % 3.0) * 4.0
            
            if min(xs) < 3 or max(xs) > self.grid_size - 8:
                self.wave_dir *= -1.0
                for e in alive:
                    e["y"] = float(e["y"]) + 2.0
            for e in alive:
                e["x"] = float(e["x"]) + self.wave_dir * 0.9 * speed_factor
                e["y"] = float(e["y"]) + math.sin(self.step_index * 0.35 + float(e["phase"])) * 0.18 * speed_factor
            if self.rng.random() < 0.16:
                shooter = alive[int(self.rng.integers(0, len(alive)))]
                self.enemy_bullets.append(
                    {"x": float(shooter["x"]) + 2, "y": float(shooter["y"]) + 4, "w": 2.0, "h": 3.0, "vy": 2.8}
                )

        hits = 0
        for bullet in self.bullets:
            bullet["x"] = float(bullet["x"]) + float(bullet.get("vx", 0.0))
            bullet["y"] = float(bullet["y"]) + float(bullet.get("vy", -4.8))
        self.bullets = [b for b in self.bullets if -6 < float(b["x"]) < self.grid_size + 6 and -6 < float(b["y"]) < self.grid_size + 6]
        for bullet in list(self.bullets):
            brect = (float(bullet["x"]), float(bullet["y"]), float(bullet["w"]), float(bullet["h"]))
            for enemy in self.enemies:
                if not bool(enemy.get("alive", True)):
                    continue
                if _rects_overlap(brect, (enemy["x"], enemy["y"], enemy["w"], enemy["h"])):
                    enemy["alive"] = False
                    hits += 1
                    self.score += 1.0
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    break

        hurt = False
        for bullet in self.enemy_bullets:
            bullet["y"] = float(bullet["y"]) + float(bullet["vy"]) * speed_factor
        self.enemy_bullets = [b for b in self.enemy_bullets if float(b["y"]) < self.grid_size + 6]
        prect = self._player_rect()
        for bullet in list(self.enemy_bullets):
            if _rects_overlap((bullet["x"], bullet["y"], bullet["w"], bullet["h"]), prect):
                self.enemy_bullets.remove(bullet)
                if self.shield_ticks > 0:
                    shielded = True
                else:
                    self.lives -= 1
                    self.score = max(0.0, self.score - 1.0)
                    hurt = True

        # Colisión directa de las naves enemigas con el jugador
        # Si un enemigo vivo colisiona con el jugador, le baja vida (castigo hedónico)
        for enemy in self.enemies:
            if bool(enemy.get("alive", True)):
                erect = (enemy["x"], enemy["y"], enemy["w"], enemy["h"])
                if _rects_overlap(erect, prect):
                    # El enemigo muere al chocar, pero daña severamente al jugador
                    enemy["alive"] = False
                    if self.shield_ticks > 0:
                        shielded = True
                    else:
                        self.lives -= 1
                        self.score = max(0.0, self.score - 1.5) # Penalización mayor por colisión física directa
                        hurt = True

        remaining = sum(1 for e in self.enemies if bool(e.get("alive", True)))
        win = remaining == 0 or self.score >= self.target_score
        dead = self.lives <= 0
        terminated = bool(win or dead)
        extrinsic = 1.0 if win else (-1.0 if dead else 0.0)
        intrinsic = -0.01 + 0.18 * hits + (0.03 if fired else 0.0) + (0.02 if shielded else 0.0) - 0.35 * int(hurt)
        sound = "win" if win else ("wrong" if hurt else ("hit" if hits else ("laser" if fired else ("bird" if alive else "step"))))
        return self._finish_step(
            reward=extrinsic,
            intrinsic_reward=float(np.clip(intrinsic, -1.0, 1.0)),
            extrinsic_reward=extrinsic,
            terminated=terminated,
            sound_kind=sound,
            sound_intensity=1.0 if win or hurt or hits else 0.45,
            info={
                "action_id": action_id,
                "action_name": self.action_name(action_id),
                "fired": int(fired),
                "shielded": int(shielded),
                "hits": int(hits),
                "collected": int(hits),
                "hazard": int(hurt),
                "collision": int(hurt),
                "remaining_targets": int(remaining),
                "score": float(self.score),
                "target_score": float(self.target_score),
                "health": int(self.lives),
                "goal_reached": int(win),
                "agent_x": int(self.player["x"]),
                "agent_y": int(self.player["y"]),
                "target_x": int(alive[0]["x"] if alive else self.grid_size // 2),
                "target_y": int(alive[0]["y"] if alive else 8),
            },
        )

    def _render_frame(self) -> np.ndarray:
        frame = _background((self.grid_size, self.grid_size), 0, accent=1, spacing=10)
        _draw_rect(frame, 0, self.grid_size - 4, self.grid_size, 4, 12)
        for enemy in self.enemies:
            if bool(enemy.get("alive", True)):
                _draw_rect(frame, enemy["x"], enemy["y"], enemy["w"], enemy["h"], int(enemy["color"]))
                _draw_rect(frame, enemy["x"] + 1, enemy["y"] - 1, max(1, enemy["w"] - 2), 1, 11)
        for bullet in self.bullets:
            _draw_rect(frame, bullet["x"], bullet["y"], bullet["w"], bullet["h"], 4)
        for bullet in self.enemy_bullets:
            _draw_rect(frame, bullet["x"], bullet["y"], bullet["w"], bullet["h"], 2)
        color = 10 if self.shield_ticks > 0 else 11
        _draw_rect(frame, self.player["x"], self.player["y"], self.player["w"], self.player["h"], color)
        _draw_rect(frame, self.player["x"] + 2, self.player["y"] - 2, 2, 2, 8)
        _draw_rect(frame, 2, 2, max(0, self.lives) * 4, 2, 3)
        _draw_rect(frame, 2, 5, int(np.clip(self.score / self.target_score, 0, 1) * (self.grid_size - 4)), 2, 4)
        return frame

    def _observation(self) -> dict[str, Any]:
        obs = super()._observation()
        alive = [e for e in self.enemies if bool(e.get("alive", True))]
        obs["agent"] = {"x": int(self.player["x"]), "y": int(self.player["y"]), "facing": 0}
        obs["target"] = {"x": int(alive[0]["x"] if alive else self.grid_size // 2), "y": int(alive[0]["y"] if alive else 8)}
        return obs


class SkyCatchGame(ArcadeGame):
    game_id = "sky_catch"
    title = "Sky Catch Signals"
    action_names = {
        1: "look_up",
        2: "duck",
        3: "move_left",
        4: "move_right",
        5: "wide_catch",
        6: "move_to_selected",
        7: "narrow_catch",
        8: "wait",
    }

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.step_index = 0
        self.action_aliases = self._make_action_aliases()
        self.player = {"x": float(self.grid_size // 2 - 5), "y": float(self.grid_size - 7), "w": 10.0, "h": 3.0}
        self.score = 0.0
        self.target_score = 12.0
        self.lives = 3
        self.catch_width_bonus = 0
        self.objects: list[dict[str, Any]] = []
        for kind in ["good", "bonus", "neutral", "bad", "deadly"]:
            self._spawn_object(start_high=True, kind=kind)
        self.last_frame = self._render_frame()
        return self._observation(), self.metadata()

    def _spawn_object(self, *, start_high: bool = False, kind: str | None = None) -> None:
        kind = str(kind or self.rng.choice(["good", "bonus", "neutral", "bad", "deadly"], p=[0.34, 0.14, 0.24, 0.18, 0.10]))
        props = {
            "good": (3, 1.0, "fall_good"),
            "bonus": (4, 2.0, "fall_good"),
            "neutral": (8, 0.0, "fall_neutral"),
            "bad": (2, -1.5, "fall_bad"),
            "deadly": (6, -3.0, "danger"),
        }[kind]
        size = float(self.rng.choice([3, 4, 5, 6]))
        self.objects.append(
            {
                "x": float(self.rng.integers(3, max(4, self.grid_size - int(size) - 3))),
                "y": float(self.rng.integers(-18, 2) if start_high else self.rng.integers(-18, -4)),
                "w": size,
                "h": size,
                "vy": float(self.rng.uniform(1.2, 2.8)),
                "kind": kind,
                "color": int(props[0]),
                "value": float(props[1]),
                "sound": str(props[2]),
            }
        )

    def physics_busy(self) -> bool:
        return True

    def accepts_busy_action(self, action_id: int) -> bool:
        return True

    def step(self, action_id: int, action_data: dict[str, Any] | None = None) -> ArcadeStepResult:
        action_id = int(action_id)
        self.step_index += 1
        speed = 4.0
        caught = 0
        neutral = 0
        penalty = 0
        deadly = 0
        if action_id == 3:
            self.player["x"] = max(1.0, float(self.player["x"]) - speed)
        elif action_id == 4:
            self.player["x"] = min(self.grid_size - float(self.player["w"]) - 1, float(self.player["x"]) + speed)
        elif action_id == 5:
            self.catch_width_bonus = 5
        elif action_id == 6:
            data = dict(action_data or {})
            tx = float(data.get("x", self.grid_size // 2))
            self.player["x"] = float(np.clip(tx - float(self.player["w"]) * 0.5, 1, self.grid_size - float(self.player["w"]) - 1))
        elif action_id == 7:
            self.catch_width_bonus = -3
        else:
            self.catch_width_bonus = max(0, int(self.catch_width_bonus) - 1)

        catch_rect = (
            float(self.player["x"]) - max(0, int(self.catch_width_bonus)),
            float(self.player["y"]) - 1,
            max(3.0, float(self.player["w"]) + max(0, int(self.catch_width_bonus)) * 2 + min(0, int(self.catch_width_bonus))),
            float(self.player["h"]) + 2,
        )
        # Progressive speed factor: starts at 0.45 and increases to 1.0 as score increases
        speed_factor = 0.45 + 0.55 * min(1.0, max(0.0, self.score) / max(1.0, self.target_score))
        
        loudest: dict[str, Any] | None = None
        for obj in list(self.objects):
            obj["y"] = float(obj["y"]) + float(obj["vy"]) * speed_factor
            if loudest is None or float(obj["y"]) > float(loudest["y"]):
                loudest = obj
            orect = (float(obj["x"]), float(obj["y"]), float(obj["w"]), float(obj["h"]))
            if _rects_overlap(orect, catch_rect):
                value = float(obj["value"])
                self.score += value
                if value > 0:
                    caught += 1
                elif value < -2.0:
                    deadly += 1
                    self.lives -= 1
                elif value < 0:
                    penalty += 1
                else:
                    neutral += 1
                self.objects.remove(obj)
                self._spawn_object()
            elif float(obj["y"]) > self.grid_size + 4:
                if str(obj["kind"]) in {"good", "bonus"}:
                    self.score -= 0.25
                self.objects.remove(obj)
                self._spawn_object()

        self.score = max(-8.0, min(20.0, float(self.score)))
        win = self.score >= self.target_score
        dead = self.lives <= 0
        extrinsic = 1.0 if win else (-1.0 if dead else 0.0)
        intrinsic = -0.01 + 0.14 * caught + 0.01 * neutral - 0.16 * penalty - 0.40 * deadly
        if loudest is None:
            sound = "step"
        elif deadly:
            sound = "danger"
        elif penalty:
            sound = "fall_bad"
        elif caught:
            sound = "score"
        else:
            visible = [o for o in self.objects if float(o.get("y", 0.0)) >= -4.0]
            visible_sounds = {str(o.get("sound", "fall_neutral")) for o in visible}
            if "danger" in visible_sounds:
                sound = "danger"
            elif "fall_bad" in visible_sounds:
                sound = "fall_bad"
            elif "fall_good" in visible_sounds:
                sound = "fall_good"
            else:
                sound = str(loudest.get("sound", "fall_neutral"))
        target_obj = max(self.objects, key=lambda o: float(o["y"]), default={"x": self.grid_size // 2, "y": 0})
        return self._finish_step(
            reward=extrinsic,
            intrinsic_reward=float(np.clip(intrinsic, -1.0, 1.0)),
            extrinsic_reward=extrinsic,
            terminated=bool(win or dead),
            sound_kind=sound,
            sound_intensity=1.0 if caught or penalty or deadly or win or dead else 0.38,
            info={
                "action_id": action_id,
                "action_name": self.action_name(action_id),
                "collected": int(caught),
                "neutral_collected": int(neutral),
                "penalty_food": int(penalty),
                "hazard": int(deadly),
                "collision": int(penalty or deadly),
                "falling_objects": int(len(self.objects)),
                "score": float(self.score),
                "target_score": float(self.target_score),
                "health": int(self.lives),
                "goal_reached": int(win),
                "agent_x": int(self.player["x"]),
                "agent_y": int(self.player["y"]),
                "target_x": int(target_obj["x"]),
                "target_y": int(target_obj["y"]),
                "fall_sound": sound,
            },
        )

    def _render_frame(self) -> np.ndarray:
        frame = _background((self.grid_size, self.grid_size), 5, accent=13, spacing=12)
        _draw_rect(frame, 0, self.grid_size - 4, self.grid_size, 4, 12)
        for obj in self.objects:
            _draw_rect(frame, obj["x"], obj["y"], obj["w"], obj["h"], int(obj["color"]))
            if str(obj["kind"]) == "deadly":
                _draw_rect(frame, obj["x"] + 1, obj["y"] + 1, max(1, obj["w"] - 2), max(1, obj["h"] - 2), 2)
        bonus = max(0, int(self.catch_width_bonus))
        _draw_rect(frame, self.player["x"] - bonus, self.player["y"], self.player["w"] + bonus * 2, self.player["h"], 11)
        _draw_rect(frame, 2, 2, max(0, self.lives) * 4, 2, 3)
        score_w = int(np.clip((self.score + 8.0) / 28.0, 0, 1) * (self.grid_size - 4))
        _draw_rect(frame, 2, 5, score_w, 2, 4)
        return frame

    def _observation(self) -> dict[str, Any]:
        obs = super()._observation()
        target_obj = max(self.objects, key=lambda o: float(o["y"]), default={"x": self.grid_size // 2, "y": 0})
        obs["agent"] = {"x": int(self.player["x"]), "y": int(self.player["y"]), "facing": 0}
        obs["target"] = {"x": int(target_obj["x"]), "y": int(target_obj["y"])}
        return obs


class DrCapsuleGame(ArcadeGame):
    game_id = "dr_capsule"
    title = "Dr Capsule Stack"
    VIRUS_RENDER_COLOR = 2
    MATCH_MIN = 4
    action_names = {
        1: "hold",
        2: "hard_drop",
        3: "move_left",
        4: "move_right",
        5: "rotate_clockwise",
        6: "move_to_selected",
        7: "rotate_counter",
        8: "wait",
    }

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.step_index = 0
        self.action_aliases = self._make_action_aliases()
        self.cols = 8
        self.rows = 14
        self.cell = max(3, self.grid_size // 16)
        self.x0 = (self.grid_size - self.cols * self.cell) // 2
        self.y0 = max(5, self.grid_size - self.rows * self.cell - 3)
        self.board = np.zeros((self.rows, self.cols), dtype=np.int16)
        self.viruses: dict[tuple[int, int], int] = {}
        self.colors = [3, 4, 8]
        self.score = 0.0
        self.combo = 0
        self.errors = 0
        self.clears = 0
        self.viruses_cleared = 0
        self.fall_clock = 0
        self.fall_interval = 24
        self.active: dict[str, Any] | None = None
        self.held: tuple[int, int] | None = None
        self.can_hold = True
        self.next_colors = self._new_colors()
        self._spawn_viruses()
        self._spawn_capsule()
        self.last_frame = self._render_frame()
        return self._observation(), self.metadata()

    def _new_colors(self) -> tuple[int, int]:
        return int(self.rng.choice(self.colors)), int(self.rng.choice(self.colors))

    def _spawn_viruses(self) -> None:
        self.viruses = {}
        # Queremos exactamente 3 virus, uno de cada color disponible
        virus_colors = list(self.colors)  # [3, 4, 8]
        self.rng.shuffle(virus_colors)
        
        # El último tercio de la botella (filas inferiores)
        start_row = int(self.rows * 2 // 3)  # Para 14 filas, start_row = 9
        
        attempts = 0
        for color in virus_colors:
            placed = False
            while not placed and attempts < 1000:
                attempts += 1
                x = int(self.rng.integers(0, self.cols))
                y = int(self.rng.integers(start_row, self.rows))
                if (x, y) in self.viruses:
                    continue
                if any(abs(x - nx) + abs(y - ny) <= 1 for nx, ny in self.viruses):
                    continue
                self.viruses[(x, y)] = color
                placed = True
                
        self.viruses_start = len(self.viruses)

    def _occupied(self, x: int, y: int) -> bool:
        return int(self.board[y, x]) != 0 or (int(x), int(y)) in self.viruses

    def _cell_color(self, x: int, y: int) -> int:
        key = (int(x), int(y))
        if key in self.viruses:
            return int(self.viruses[key])
        return int(self.board[y, x])

    def _capsule_cells(self, cap: dict[str, Any] | None = None) -> list[tuple[int, int, int]]:
        cap = cap or self.active
        if cap is None:
            return []
        x = int(cap["x"])
        y = int(cap["y"])
        orientation = int(cap.get("orientation", 0)) % 4
        c0, c1 = int(cap["colors"][0]), int(cap["colors"][1])
        if orientation == 0:
            return [(x, y, c0), (x + 1, y, c1)]
        elif orientation == 1:
            return [(x, y, c0), (x, y + 1, c1)]
        elif orientation == 2:
            return [(x, y, c1), (x + 1, y, c0)] # Colores invertidos horizontalmente
        else:
            return [(x, y, c1), (x, y + 1, c0)] # Colores invertidos verticalmente

    def _valid_capsule(self, cap: dict[str, Any]) -> bool:
        for x, y, _color in self._capsule_cells(cap):
            if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
                return False
            if self._occupied(x, y):
                return False
        return True

    def _spawn_capsule(self) -> bool:
        colors = self.next_colors
        self.next_colors = self._new_colors()
        cap = {"x": self.cols // 2 - 1, "y": 0, "orientation": 0, "colors": colors}
        if not self._valid_capsule(cap):
            self.active = None
            return False
        self.active = cap
        self.fall_clock = 0
        self.can_hold = True
        return True

    def _try_hold(self) -> bool:
        if self.active is None or not self.can_hold:
            return False
        current = tuple(int(c) for c in self.active["colors"])
        spawn = {"x": self.cols // 2 - 1, "y": 0, "orientation": 0}
        if self.held is None:
            candidate = {**spawn, "colors": self.next_colors}
            if not self._valid_capsule(candidate):
                return False
            self.held = current
            self.active = candidate
            self.next_colors = self._new_colors()
        else:
            candidate = {**spawn, "colors": self.held}
            if not self._valid_capsule(candidate):
                return False
            self.held = current
            self.active = candidate
        self.can_hold = False
        self.fall_clock = 0
        return True

    def _try_move_active(self, dx: int, dy: int) -> bool:
        if self.active is None:
            return False
        cap = dict(self.active)
        cap["x"] = int(cap["x"]) + int(dx)
        cap["y"] = int(cap["y"]) + int(dy)
        if not self._valid_capsule(cap):
            return False
        self.active = cap
        return True

    def _try_rotate(self, delta: int) -> bool:
        if self.active is None:
            return False
        cap = dict(self.active)
        cap["orientation"] = (int(cap.get("orientation", 0)) + int(delta)) % 4
        candidates = [cap]
        for kick in (-1, 1, -2, 2):
            kicked = dict(cap)
            kicked["x"] = int(kicked["x"]) + kick
            candidates.append(kicked)
        for candidate in candidates:
            if self._valid_capsule(candidate):
                self.active = candidate
                return True
        return False

    def _settle_active(self) -> tuple[int, int, int, int, bool]:
        if self.active is None:
            return 0, 0, 0, 0, False
        placed = self._capsule_cells(self.active)
        for x, y, color in placed:
            self.board[y, x] = int(color)
        same_pairs, mismatch_pairs = self._adjacency_quality(placed)
        cleared, viruses_removed = self._resolve_matches_and_gravity()
        self.viruses_cleared += int(viruses_removed)
        overflow = bool(np.any(self.board[0:2, :] != 0))
        for sx, sy in ((self.cols // 2 - 1, 0), (self.cols // 2, 0), (self.cols // 2 - 1, 1), (self.cols // 2, 1)):
            if 0 <= sx < self.cols and 0 <= sy < self.rows and self._occupied(sx, sy):
                overflow = True
                break
        spawned = self._spawn_capsule()
        return same_pairs, mismatch_pairs, cleared, viruses_removed, bool(overflow or not spawned)

    def _adjacency_quality(self, placed: list[tuple[int, int, int]]) -> tuple[int, int]:
        same = 0
        mismatch = 0
        seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for x, y, color in placed:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= self.cols or ny < 0 or ny >= self.rows:
                    continue
                other = self._cell_color(nx, ny)
                if other == 0:
                    continue
                key = tuple(sorted(((x, y), (nx, ny))))
                if key in seen:
                    continue
                seen.add(key)
                if other == int(color):
                    same += 1
                else:
                    mismatch += 1
        return same, mismatch

    def _find_matches(self) -> set[tuple[int, int]]:
        matched: set[tuple[int, int]] = set()
        min_len = int(self.MATCH_MIN)
        for y in range(self.rows):
            x = 0
            while x < self.cols:
                color = self._cell_color(x, y)
                if color == 0:
                    x += 1
                    continue
                x2 = x
                while x2 < self.cols and self._cell_color(x2, y) == color:
                    x2 += 1
                if x2 - x >= min_len:
                    for xx in range(x, x2):
                        matched.add((xx, y))
                x = x2
        for x in range(self.cols):
            y = 0
            while y < self.rows:
                color = self._cell_color(x, y)
                if color == 0:
                    y += 1
                    continue
                y2 = y
                while y2 < self.rows and self._cell_color(x, y2) == color:
                    y2 += 1
                if y2 - y >= min_len:
                    for yy in range(y, y2):
                        matched.add((x, yy))
                y = y2
        return matched

    def _apply_gravity(self) -> bool:
        moved = False
        for x in range(self.cols):
            y = self.rows - 2
            while y >= 0:
                if (x, y) in self.viruses or int(self.board[y, x]) == 0:
                    y -= 1
                    continue
                drop_to = y
                ny = y + 1
                while ny < self.rows:
                    if (x, ny) in self.viruses:
                        break
                    if int(self.board[ny, x]) != 0:
                        break
                    drop_to = ny
                    ny += 1
                if drop_to > y:
                    self.board[drop_to, x] = int(self.board[y, x])
                    self.board[y, x] = 0
                    moved = True
                y -= 1
        return moved

    def _resolve_matches_and_gravity(self) -> tuple[int, int]:
        total = 0
        viruses_removed = 0
        for _ in range(8):
            matched = self._find_matches()
            if not matched:
                break
            total += len(matched)
            for x, y in matched:
                key = (int(x), int(y))
                if key in self.viruses:
                    del self.viruses[key]
                    viruses_removed += 1
                self.board[y, x] = 0
            self._apply_gravity()
        return total, viruses_removed

    def _hard_drop_active(self) -> bool:
        if self.active is None:
            return False
        dropped = False
        while self._try_move_active(0, 1):
            dropped = True
        return dropped

    def physics_busy(self) -> bool:
        return True

    def accepts_busy_action(self, action_id: int) -> bool:
        return True

    def step(self, action_id: int, action_data: dict[str, Any] | None = None) -> ArcadeStepResult:
        action_id = int(action_id)
        self.step_index += 1
        moved = False
        rotated = False
        held = False
        hard_drop = False
        if action_id == 1:
            held = self._try_hold()
        elif action_id == 2:
            hard_drop = self._hard_drop_active()
            moved = hard_drop
        elif action_id == 3:
            moved = self._try_move_active(-1, 0)
        elif action_id == 4:
            moved = self._try_move_active(1, 0)
        elif action_id == 5:
            rotated = self._try_rotate(1)
        elif action_id == 6:
            data = dict(action_data or {})
            tx = int(float(data.get("x", self.grid_size // 2)))
            bottle_x = int((tx - self.x0) // max(1, self.cell))
            if self.active is not None:
                moved = self._try_move_active(int(np.sign(bottle_x - int(self.active["x"]))), 0)
        elif action_id == 7:
            rotated = self._try_rotate(-1)

        # Progressive speed increase for DrCapsuleGame:
        # Start slower (fall_interval = 24, falls every 12 steps)
        # Speed up as viruses are cleared or steps pass
        self.fall_interval = max(6, 24 - self.viruses_cleared * 2 - self.step_index // 100)

        if action_id != 2:
            self.fall_clock += 1
        settled = False
        same_pairs = 0
        mismatch_pairs = 0
        cleared = 0
        viruses_removed = 0
        lost = False
        if self.active is not None:
            if action_id == 2:
                settled = True
                same_pairs, mismatch_pairs, cleared, viruses_removed, lost = self._settle_active()
            elif self.fall_clock >= self.fall_interval:
                self.fall_clock = 0
                if not self._try_move_active(0, 1):
                    settled = True
                    same_pairs, mismatch_pairs, cleared, viruses_removed, lost = self._settle_active()

        self.combo = self.combo + 1 if cleared > 0 else 0
        self.errors += int(mismatch_pairs > same_pairs and settled)
        self.clears += int(cleared > 0)
        viruses_remaining = len(self.viruses)
        won = bool(viruses_remaining == 0 and not lost)
        extrinsic = 1.0 if won else (-1.0 if lost else 0.0)
        match_reward = 0.018 * same_pairs
        clear_reward = 0.13 * cleared * (1.0 + 0.15 * self.combo)
        virus_reward = 0.35 * viruses_removed
        mismatch_penalty = 0.014 * mismatch_pairs
        intrinsic = (
            -0.004
            + match_reward
            + clear_reward
            + virus_reward
            - mismatch_penalty
            - 0.75 * int(lost)
            + 0.5 * int(won)
        )
        self.score += max(0.0, match_reward + clear_reward + virus_reward) - mismatch_penalty
        sound = "win" if won else (
            "wrong"
            if lost
            else (
                "bonus"
                if viruses_removed
                else (
                    "bonus"
                    if cleared
                    else (
                        "score"
                        if same_pairs
                        else ("collision" if mismatch_pairs else ("slide" if moved or rotated or held or hard_drop else "step"))
                    )
                )
            )
        )
        target_x, target_y = self._virus_target_cell()
        return self._finish_step(
            reward=extrinsic,
            intrinsic_reward=float(np.clip(intrinsic, -1.0, 1.0)),
            extrinsic_reward=extrinsic,
            terminated=bool(won or lost),
            sound_kind=sound,
            sound_intensity=1.0 if won or lost or cleared or viruses_removed else 0.42,
            info={
                "action_id": action_id,
                "action_name": self.action_name(action_id),
                "capsule_moved": int(moved),
                "capsule_rotated": int(rotated),
                "capsule_held": int(held),
                "hard_drop": int(hard_drop),
                "capsule_settled": int(settled),
                "same_color_pairs": int(same_pairs),
                "mismatch_pairs": int(mismatch_pairs),
                "pieces_cleared": int(cleared),
                "viruses_cleared": int(viruses_removed),
                "viruses_remaining": int(viruses_remaining),
                "viruses_start": int(getattr(self, "viruses_start", viruses_remaining)),
                "combo": int(self.combo),
                "errors": int(self.errors),
                "clears": int(self.clears),
                "score": float(self.score),
                "target_score": float(getattr(self, "viruses_start", viruses_remaining)),
                "hazard": int(lost),
                "collision": int(mismatch_pairs > 0 or lost),
                "goal_reached": int(won),
                "agent_x": int(self.x0 + (int(self.active["x"]) if self.active else self.cols // 2) * self.cell),
                "agent_y": int(self.y0 + (int(self.active["y"]) if self.active else 0) * self.cell),
                "target_x": int(self.x0 + target_x * self.cell),
                "target_y": int(self.y0 + target_y * self.cell),
            },
        )

    def _virus_target_cell(self) -> tuple[int, int]:
        if not self.viruses:
            return self.cols // 2, self.rows // 2
        return min(self.viruses.keys(), key=lambda p: (p[1], abs(p[0] - self.cols // 2)))

    def _render_frame(self) -> np.ndarray:
        frame = _background((self.grid_size, self.grid_size), 0, accent=5, spacing=8)
        _draw_border(frame, self.x0 - 2, self.y0 - 2, self.cols * self.cell + 4, self.rows * self.cell + 4, 13, thickness=2)
        for y in range(self.rows):
            for x in range(self.cols):
                color = int(self.board[y, x])
                if color:
                    _draw_rect(frame, self.x0 + x * self.cell, self.y0 + y * self.cell, self.cell - 1, self.cell - 1, color)
        virus_r = max(1.0, (self.cell - 1) * 0.42)
        for (vx, vy), _vcolor in self.viruses.items():
            cx = self.x0 + vx * self.cell + (self.cell - 1) * 0.5
            cy = self.y0 + vy * self.cell + (self.cell - 1) * 0.5
            # Dibujamos el virus con su color real de match (para que meha entienda la alineación)
            # y le añadimos un pequeño centro blanco brillante para denotar que es un virus
            _draw_filled_circle(frame, cx, cy, virus_r, _vcolor)
            _draw_filled_circle(frame, cx, cy, max(0.5, virus_r * 0.35), 11)
        if self.active is not None:
            for x, y, color in self._capsule_cells(self.active):
                _draw_rect(frame, self.x0 + x * self.cell, self.y0 + y * self.cell, self.cell - 1, self.cell - 1, color)
        n0, n1 = self.next_colors
        _draw_rect(frame, 2, 2, self.cell, self.cell, n0)
        _draw_rect(frame, 2 + self.cell, 2, self.cell, self.cell, n1)
        if self.held is not None:
            h0, h1 = self.held
            _draw_rect(frame, 2, 2 + self.cell + 2, self.cell, self.cell, h0)
            _draw_rect(frame, 2 + self.cell, 2 + self.cell + 2, self.cell, self.cell, h1)
            _draw_border(frame, 2, 2 + self.cell + 2, self.cell * 2, self.cell, 11, thickness=1)
        virus_w = int(
            np.clip(len(self.viruses) / max(1, getattr(self, "viruses_start", len(self.viruses) or 1)), 0, 1)
            * (self.grid_size - 4)
        )
        _draw_rect(frame, 2, self.grid_size - 3, virus_w, 2, 2)
        return frame

    def _observation(self) -> dict[str, Any]:
        obs = super()._observation()
        ax = self.x0 + (int(self.active["x"]) if self.active else self.cols // 2) * self.cell
        ay = self.y0 + (int(self.active["y"]) if self.active else 0) * self.cell
        obs["agent"] = {"x": int(ax), "y": int(ay), "facing": int(self.active.get("orientation", 0)) if self.active else 0}
        tx, ty = self._virus_target_cell()
        obs["target"] = {"x": int(self.x0 + tx * self.cell), "y": int(self.y0 + ty * self.cell)}
        return obs


class FabulousFredGame(ArcadeGame):
    game_id = "fabulous_fred"
    title = "Fabulous Fred Memory"
    action_names = {
        1: "wait",
        2: "wait",
        3: "wait",
        4: "wait",
        5: "wait",
        6: "touch_selected",
        7: "wait",
        8: "wait",
    }

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.step_index = 0
        self.action_aliases = self._make_action_aliases()
        
        # Game parameters
        self.sequence_len = 3
        self.pause_time = 1.5  # seconds between each step of the sequence (reduced from 3.0)
        self.turn = "machine"  # "machine" or "player"
        
        # Generate initial sequence of length 3
        self.sequence = [int(self.rng.integers(0, 6)) for _ in range(self.sequence_len)]
        
        # Machine turn state
        self.machine_seq_index = 0
        self.machine_sub_state = "lit"  # "lit" or "pause"
        self.machine_timer = 10  # 1.0 second (10 steps) to light up the first square
        
        # Player turn state
        self.player_seq_index = 0
        self.player_timer = 200  # 20.0 seconds (200 steps) total for player's turn
        
        # Feedback animation state for player touches
        self.feedback_timer = 0
        self.feedback_square = -1
        self.feedback_type = None
        
        # Square definitions (6 squares in a 2x3 grid)
        # Each square: {"dark_color": int, "bright_color": int, "sound": str, "x": int, "y": int, "w": int, "h": int}
        self.squares = [
            {"dark_color": 1, "bright_color": 8, "sound": "laser", "x": 8, "y": 6, "w": 20, "h": 14},       # Row 0, Col 0 (Blue)
            {"dark_color": 9, "bright_color": 2, "sound": "bird", "x": 36, "y": 6, "w": 20, "h": 14},       # Row 0, Col 1 (Red)
            {"dark_color": 3, "bright_color": 10, "sound": "jump", "x": 8, "y": 24, "w": 20, "h": 14},      # Row 1, Col 0 (Green)
            {"dark_color": 12, "bright_color": 4, "sound": "score", "x": 36, "y": 24, "w": 20, "h": 14},     # Row 1, Col 1 (Yellow)
            {"dark_color": 15, "bright_color": 6, "sound": "launch", "x": 8, "y": 42, "w": 20, "h": 14},    # Row 2, Col 0 (Magenta)
            {"dark_color": 12, "bright_color": 7, "sound": "hit", "x": 36, "y": 42, "w": 20, "h": 14},       # Row 2, Col 1 (Orange)
        ]
        
        self.last_frame = self._render_frame()
        return self._observation(), self.metadata()

    def _get_square_at(self, x: float, y: float) -> int | None:
        for idx, sq in enumerate(self.squares):
            if sq["x"] <= x < sq["x"] + sq["w"] and sq["y"] <= y < sq["y"] + sq["h"]:
                return idx
        return None

    def physics_busy(self) -> bool:
        # Physics is busy during the machine's turn or during the player's feedback animation
        return self.turn == "machine" or self.feedback_timer > 0

    def accepts_busy_action(self, action_id: int) -> bool:
        # Only wait (action 8) is allowed when physics is busy
        return int(action_id) == 8

    def step(self, action_id: int, action_data: dict[str, Any] | None = None) -> ArcadeStepResult:
        action_id = int(action_id)
        if self.turn == "player" and self.feedback_timer == 0:
            self.step_index += 1
        
        # Default values
        extrinsic = 0.0
        intrinsic = 0.0
        terminated = False
        sound = "silence"
        sound_intensity = 0.0
        blocked = bool(self.physics_busy() and not self.accepts_busy_action(action_id))
        
        if blocked:
            action_id = 8
            
        # 1. Handle feedback animation timer
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
            if self.feedback_timer == 0:
                # Feedback animation ended! Apply the pending transition
                if self.feedback_type == "round_win":
                    self.sequence_len += 1
                    self.pause_time = max(1.0, self.pause_time - 0.25)
                    self.turn = "machine"
                    self.sequence = [int(self.rng.integers(0, 6)) for _ in range(self.sequence_len)]
                    self.machine_seq_index = 0
                    self.machine_sub_state = "lit"
                    self.machine_timer = 10
                elif self.feedback_type == "game_win":
                    terminated = True
                elif self.feedback_type in ("wrong_touch", "timeout"):
                    # Reset to start
                    self.sequence_len = 3
                    self.pause_time = 1.5
                    self.turn = "machine"
                    self.sequence = [int(self.rng.integers(0, 6)) for _ in range(self.sequence_len)]
                    self.machine_seq_index = 0
                    self.machine_sub_state = "lit"
                    self.machine_timer = 10
                    self.player_seq_index = 0
                    self.player_timer = 200
                
                # Clear feedback state
                self.feedback_square = -1
                self.feedback_type = None
                
        # 2. Handle turns
        if self.turn == "machine":
            # Machine's turn is driven by wait actions (or auto-physics steps)
            self.machine_timer -= 1
            
            # Sound kind during lit sub-state
            if self.machine_sub_state == "lit" and self.machine_timer == 9:
                # Play the sound of the currently lit square at the start of its lit period
                current_sq_idx = self.sequence[self.machine_seq_index]
                sound = self.squares[current_sq_idx]["sound"]
                sound_intensity = 1.0
                
            if self.machine_timer <= 0:
                if self.machine_sub_state == "lit":
                    self.machine_sub_state = "pause"
                    self.machine_timer = int(self.pause_time * 10)
                else:  # "pause"
                    self.machine_seq_index += 1
                    if self.machine_seq_index >= len(self.sequence):
                        # Transition to player's turn
                        self.turn = "player"
                        self.player_seq_index = 0
                        self.player_timer = 200  # 20 seconds
                    else:
                        self.machine_sub_state = "lit"
                        self.machine_timer = 10  # 1.0 second
                        
        elif self.turn == "player":
            # Player's turn: decrement timer only when NOT in feedback animation
            if self.feedback_timer == 0:
                self.player_timer -= 1
                
                if self.player_timer <= 0:
                    # Timeout! (No terminamos el juego, solo reseteamos al inicio de longitud 3 tras la animación)
                    intrinsic = -0.50
                    sound = "wrong"
                    sound_intensity = 1.0
                    
                    self.feedback_timer = 8  # 0.8 seconds of failure flash
                    self.feedback_square = -1
                    self.feedback_type = "timeout"
                    
                elif action_id == 6:  # touch_selected
                    data = dict(action_data or {})
                    x = float(data.get("x", self.grid_size // 2))
                    y = float(data.get("y", self.grid_size // 2))
                    
                    sq = self._get_square_at(x, y)
                    target_sq = self.sequence[self.player_seq_index]
                    
                    if sq is not None:
                        # Click inside a rectangle!
                        if sq == target_sq:
                            # Correct touch!
                            self.player_seq_index += 1
                            intrinsic = 0.15
                            sound = self.squares[sq]["sound"]
                            sound_intensity = 1.0
                            
                            # Check if sequence is completed
                            if self.player_seq_index >= len(self.sequence):
                                # Sequence completed successfully!
                                if self.sequence_len + 1 >= 8:
                                    # Complete game win!
                                    extrinsic = 1.0
                                    self.feedback_timer = 12  # 1.2 seconds of victory fanfare!
                                    self.feedback_square = sq
                                    self.feedback_type = "game_win"
                                    sound = "win"
                                    sound_intensity = 1.0
                                else:
                                    # Round completed!
                                    extrinsic = 0.50  # Partial reward
                                    self.feedback_timer = 8  # 0.8 seconds of round victory fanfare!
                                    self.feedback_square = sq
                                    self.feedback_type = "round_win"
                                    sound = "win"
                                    sound_intensity = 0.8
                            else:
                                # Correct touch but sequence not yet finished
                                self.feedback_timer = 4  # 0.4 seconds of feedback
                                self.feedback_square = sq
                                self.feedback_type = "correct"
                        else:
                            # Incorrect touch inside a rectangle (This IS a failure/fallo)
                            intrinsic = -0.50
                            sound = "wrong"
                            sound_intensity = 1.0
                            
                            self.feedback_timer = 8  # 0.8 seconds of failure flash!
                            self.feedback_square = sq  # Show the wrong square they clicked
                            self.feedback_type = "wrong_touch"
                    else:
                        # Incorrect touch OUTSIDE any rectangle (NOT a failure, just a penalty, turn continues)
                        intrinsic = -0.25  # Negative reward so MEHA knows it's wrong
                        sound = "wrong"
                        sound_intensity = 0.5  # Lower intensity sound to distinguish it
                        
                        self.feedback_timer = 3  # 0.3 seconds of feedback
                        self.feedback_square = -1
                        self.feedback_type = "outside_touch"
                        
        return self._finish_step(
            reward=extrinsic,
            intrinsic_reward=float(np.clip(intrinsic, -1.0, 1.0)),
            extrinsic_reward=extrinsic,
            terminated=terminated,
            sound_kind=sound,
            sound_intensity=sound_intensity,
            info={
                "action_id": action_id,
                "action_name": self.action_name(action_id),
                "action_blocked": int(blocked),
                "physics_busy": int(self.physics_busy()),
                "turn": self.turn,
                "sequence_len": int(self.sequence_len),
                "player_seq_index": int(self.player_seq_index),
                "player_timer": int(self.player_timer),
                "pause_time": float(self.pause_time),
                "correct_touch": int(sq is not None and sq == target_sq if 'sq' in locals() and 'target_sq' in locals() else 0),
                "wrong_touch": int(sq is None or sq != target_sq if 'sq' in locals() and 'target_sq' in locals() else 0),
                "timeout": int(self.player_timer <= 0),
                "round_completed": int(self.player_seq_index >= len(self.sequence) if self.turn == "player" else 0),
            }
        )

    def _render_frame(self) -> np.ndarray:
        # Background
        frame = _background((self.grid_size, self.grid_size), 0, accent=5, spacing=16)
        
        # Draw a border around the play area
        border_color = 13  # Default dark blue/purple
        if self.feedback_timer > 0:
            if self.feedback_type in ("round_win", "game_win"):
                # Flash border between green (3) and white (11)
                border_color = 3 if (self.feedback_timer // 2) % 2 == 0 else 11
            elif self.feedback_type in ("wrong_touch", "timeout"):
                # Flash border between bright red (2) and dark red (9)
                border_color = 2 if (self.feedback_timer // 2) % 2 == 0 else 9
                
        _draw_border(frame, 2, 2, self.grid_size - 4, self.grid_size - 4, border_color, thickness=2)
        
        # Draw the 6 squares
        for idx, sq in enumerate(self.squares):
            # Check if this square is lit
            is_lit = False
            is_wrong_lit = False
            if self.turn == "machine" and self.machine_sub_state == "lit" and self.sequence[self.machine_seq_index] == idx:
                is_lit = True
            elif self.turn == "player" and self.feedback_timer > 0:
                if self.feedback_type in ("round_win", "game_win"):
                    # Flashing fanfare: all squares flash together
                    is_lit = ((self.feedback_timer // 2) % 2 == 0)
                elif self.feedback_type == "wrong_touch" and self.feedback_square == idx:
                    # Flash red for wrong touch
                    is_wrong_lit = ((self.feedback_timer // 2) % 2 == 0)
                elif self.feedback_square == idx:
                    is_lit = True
                
            if is_wrong_lit:
                # Flash red for wrong touch
                _draw_rect(frame, sq["x"], sq["y"], sq["w"], sq["h"], 2)  # Bright Red
                _draw_border(frame, sq["x"], sq["y"], sq["w"], sq["h"], 11, thickness=2)
            elif is_lit:
                # Lit square: filled with bright_color + white border
                _draw_rect(frame, sq["x"], sq["y"], sq["w"], sq["h"], sq["bright_color"])
                _draw_border(frame, sq["x"], sq["y"], sq["w"], sq["h"], 11, thickness=2)
            else:
                # Unlit square: filled with dark_color + bright_color border
                _draw_rect(frame, sq["x"] + 2, sq["y"] + 2, sq["w"] - 4, sq["h"] - 4, sq["dark_color"])
                _draw_border(frame, sq["x"], sq["y"], sq["w"], sq["h"], sq["bright_color"], thickness=2)
                
        # Draw turn indicators at the top
        if self.turn == "machine":
            # Blue circle on top left
            _draw_filled_circle(frame, 4, 3, 2, 1)
        elif self.turn == "player":
            # Green circle on top right
            _draw_filled_circle(frame, self.grid_size - 4, 3, 2, 3)
            
            # Draw player's remaining time progress bar at the bottom
            bar_w = int(np.clip(self.player_timer / 200.0, 0, 1) * (self.grid_size - 8))
            _draw_rect(frame, 4, self.grid_size - 4, bar_w, 2, 4)  # Yellow/Green bar
            
        return frame

    def _observation(self) -> dict[str, Any]:
        obs = super()._observation()
        
        # Grounding agent and target coordinates
        # Agent position is the selected cell (or center if not selected)
        # Target position is the square that the player needs to touch next
        target_idx = self.sequence[self.player_seq_index] if self.turn == "player" and self.player_seq_index < len(self.sequence) else 0
        target_sq = self.squares[target_idx]
        
        obs["agent"] = {"x": self.grid_size // 2, "y": self.grid_size // 2, "facing": 0}
        obs["target"] = {"x": int(target_sq["x"] + target_sq["w"] // 2), "y": int(target_sq["y"] + target_sq["h"] // 2)}
        return obs


GAME_CLASSES: dict[str, type[ArcadeGame]] = {
    "p61_sort": ColorSortGame,
    "nibbles": NibblesGame,
    "angry_blocks": AngryBlocksGame,
    "platformer": PlatformerGame,
    "phoenix_duel": PhoenixDuelGame,
    "sky_catch": SkyCatchGame,
    "dr_capsule": DrCapsuleGame,
    "fabulous_fred": FabulousFredGame,
}


class ArcadeSuiteEnv:
    """Randomized multi-game wrapper for human demonstrations."""

    def __init__(self, *, grid_size: int = 64, seed: int = 0, games: list[str] | None = None) -> None:
        self.grid_size = int(grid_size)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        requested = games or list(GAME_CLASSES.keys())
        self.games = [g for g in requested if g in GAME_CLASSES]
        if not self.games:
            self.games = list(GAME_CLASSES.keys())
        self.current_game: ArcadeGame | None = None
        self.current_game_id = ""
        self.episode_index = 0

    def reset(self, *, seed: int | None = None, game_id: str | None = None, avoid_game_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)
        candidates = list(self.games)
        if avoid_game_id and len(candidates) > 1:
            candidates = [g for g in candidates if g != avoid_game_id] or candidates
        chosen = str(game_id) if game_id in GAME_CLASSES else str(self.rng.choice(candidates))
        game_seed = int(self.rng.integers(0, 2**31 - 1))
        cls = GAME_CLASSES[chosen]
        self.current_game = cls(grid_size=self.grid_size, seed=game_seed)
        self.current_game_id = chosen
        self.episode_index += 1
        obs, info = self.current_game.reset(seed=game_seed)
        info = dict(info)
        info.update({"suite_seed": int(self.seed), "suite_episode": int(self.episode_index)})
        return obs, info

    def step(self, action_id: int, action_data: dict[str, Any] | None = None) -> ArcadeStepResult:
        if self.current_game is None:
            self.reset(seed=self.seed)
        assert self.current_game is not None
        return self.current_game.step(int(action_id), dict(action_data or {}))

    def action_name(self, action_id: int) -> str:
        if self.current_game is None:
            return ACTION_LABELS.get(int(action_id), "unknown")
        return self.current_game.action_name(int(action_id))

    def accepts_busy_action(self, action_id: int) -> bool:
        if self.current_game is None:
            return int(action_id) == 8
        return bool(self.current_game.accepts_busy_action(int(action_id)))

    def metadata(self) -> dict[str, Any]:
        if self.current_game is None:
            return {"game_id": "", "title": "", "available_games": self.games}
        meta = self.current_game.metadata()
        meta["available_games"] = list(self.games)
        return meta

    def physics_busy(self) -> bool:
        if self.current_game is None:
            return False
        return bool(self.current_game.physics_busy())

    def render_minimap(self, *, size: int = 128) -> list[list[int]]:
        if self.current_game is None:
            return np.zeros((size, size), dtype=int).tolist()
        return self.current_game.render_minimap(size=size)

    @property
    def step_index(self) -> int:
        if self.current_game is None:
            return 0
        return int(self.current_game.step_index)
