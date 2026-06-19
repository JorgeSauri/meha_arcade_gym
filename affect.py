from __future__ import annotations

import math
from typing import Any


POSITIVE_EVENTS = {"score", "bonus", "win", "fall_good"}
NEGATIVE_EVENTS = {"wrong", "collision", "penalty", "fall_bad", "danger"}
MOTION_EVENTS = {"step", "slide", "push", "jump", "launch", "hit", "laser", "bird", "fall_neutral"}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp11(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _f(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out != out:
        return float(default)
    return float(out)


def normalized_action_entropy(action_history: list[int], *, num_actions: int = 8) -> float:
    actions = [int(a) for a in action_history if 1 <= int(a) <= int(num_actions)]
    if not actions:
        return 0.0
    counts = [0] * int(num_actions)
    for action in actions:
        counts[action - 1] += 1
    total = float(len(actions))
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = float(count) / total
        entropy -= p * math.log(p)
    return _clamp01(entropy / math.log(max(2, int(num_actions))))


def affect_for_transition(
    *,
    info: dict[str, Any],
    reward: dict[str, Any],
    action_history: list[int],
    sound_event: str,
    num_actions: int = 8,
) -> dict[str, Any]:
    """Dense affect signals aligned with MEHA V7 VAN/limbic conventions.

    These are environment-side observation labels for human demos. They are not
    MEHA policy outputs; entropy is estimated from recent human action diversity.
    """

    info = dict(info or {})
    reward = dict(reward or {})
    sound_event = str(sound_event or info.get("sound_event", "silence") or "silence")

    train = _clamp11(_f(reward.get("train"), 0.0))
    extrinsic = _f(reward.get("extrinsic"), _f(info.get("extrinsic_reward"), 0.0))
    intrinsic = _f(reward.get("intrinsic"), _f(info.get("intrinsic_reward"), 0.0))
    changed_ratio = _clamp01(_f(info.get("changed_ratio"), 0.0))
    salience = _clamp01(changed_ratio / 0.12)

    positive_event = bool(
        sound_event in POSITIVE_EVENTS
        or extrinsic > 0.0
        or _f(info.get("object_delivered"), 0.0) > 0.0
        or _f(info.get("collected"), 0.0) > 0.0
        or _f(info.get("target_hits"), 0.0) > 0.0
        or bool(info.get("goal_reached", False))
    )
    negative_event = bool(
        sound_event in NEGATIVE_EVENTS
        or _f(info.get("collision"), 0.0) > 0.0
        or _f(info.get("hazard"), 0.0) > 0.0
        or _f(info.get("penalty_food"), 0.0) > 0.0
        or bool(info.get("no_op", False))
        or bool(info.get("action_blocked", False))
    )
    motor_event = bool(
        sound_event in MOTION_EVENTS
        or _f(info.get("object_moved"), 0.0) > 0.0
        or _f(info.get("hits"), 0.0) > 0.0
        or _f(info.get("jumped"), 0.0) > 0.0
        or _f(info.get("launch"), 0.0) > 0.0
    )

    event_strength = _clamp01(
        0.35 * float(positive_event)
        + 0.35 * float(negative_event)
        + 0.20 * float(motor_event)
        + 0.30 * abs(train)
    )
    novelty = _clamp01(max(salience, 0.55 * event_strength + 0.45 * salience))
    entropy = normalized_action_entropy(action_history[-32:], num_actions=int(num_actions))
    van_trigger_strength = _clamp01(max(salience, novelty, entropy))
    arousal = _clamp01(0.40 * salience + 0.30 * entropy + 0.20 * novelty + 0.10 * van_trigger_strength)
    approach = _clamp01(max(0.0, train) + 0.25 * float(positive_event) + 0.20 * novelty)
    avoidance = _clamp01(max(0.0, -train) + 0.30 * float(negative_event) + 0.25 * float(bool(info.get("no_op", False))))
    curiosity = _clamp01(0.45 * novelty + 0.35 * entropy + 0.20 * salience)
    priority = _clamp01(0.45 * arousal + 0.30 * curiosity + 0.25 * max(approach, avoidance))

    tags: list[str] = []
    if positive_event:
        tags.append("positive_event")
    if negative_event:
        tags.append("negative_event")
    if motor_event:
        tags.append("motor_event")
    if sound_event and sound_event != "silence":
        tags.append(f"sound:{sound_event}")

    return {
        "schema": "meha_human_affect_v1",
        "valence": train,
        "salience": salience,
        "novelty": novelty,
        "entropy": entropy,
        "arousal": arousal,
        "approach": approach,
        "avoidance": avoidance,
        "curiosity": curiosity,
        "priority": priority,
        "van_trigger_strength": van_trigger_strength,
        "positive_event": int(positive_event),
        "negative_event": int(negative_event),
        "motor_event": int(motor_event),
        "event_strength": event_strength,
        "changed_ratio": changed_ratio,
        "sound_event": sound_event,
        "event_tags": tags,
        "entropy_source": "recent_human_action_distribution",
        "entropy_window": min(len(action_history), 32),
        "num_actions": int(num_actions),
        "notes": "Environment-side affect labels for observation learning; entropy is not MEHA policy entropy.",
    }
