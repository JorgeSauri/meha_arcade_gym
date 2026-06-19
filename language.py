from __future__ import annotations

from typing import Any


COLOR_WORDS = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "gray",
    6: "magenta",
    7: "orange",
    8: "cyan",
    9: "brown",
    10: "lime",
    11: "white",
    12: "dark-brown",
    13: "silver",
    14: "turquoise",
    15: "purple",
}


ACTION_WORDS = {
    1: ["up", "north"],
    2: ["down", "south"],
    3: ["left", "west"],
    4: ["right", "east"],
    5: ["primary"],
    6: ["targeted", "local"],
    7: ["secondary"],
    8: ["wait"],
}


def _num(value: object, *, scale: float = 1.0, digits: int = 1) -> str:
    try:
        return str(round(float(value) * float(scale), int(digits)))
    except (TypeError, ValueError):
        return "0"


def _base_tokens(action_id: int, action_name: str, info: dict[str, Any]) -> list[str]:
    tokens = ["game", str(info.get("game_id", "arcade"))]
    tokens.extend(ACTION_WORDS.get(int(action_id), ["action"]))
    tokens.extend(str(action_name or "").replace("/", " ").replace("_", " ").split())
    if info.get("physics_busy"):
        tokens.extend(["physics", "gravity", "motion"])
    if info.get("action_blocked"):
        tokens.extend(["input", "blocked"])
    if info.get("collision"):
        tokens.extend(["collision", "contact"])
    if info.get("no_op"):
        tokens.extend(["no-op", "still"])
    sound = str(info.get("sound_event", "silence"))
    if sound and sound != "silence":
        tokens.extend(["sound", sound])
    return tokens


def language_for_transition(
    *,
    game_id: str,
    action_id: int,
    action_name: str,
    action_key: str,
    action_data: dict[str, Any] | None,
    info: dict[str, Any],
    reward: dict[str, Any],
    max_tokens: int = 48,
) -> dict[str, Any]:
    info = dict(info or {})
    reward = dict(reward or {})
    tokens = _base_tokens(int(action_id), str(action_name), info)
    text_parts: list[str] = []

    if game_id == "p61_sort":
        tokens.extend(["agent", "red", "cursor", "push"])
        if info.get("object_moved"):
            tokens.extend(["moving", "colored", "block", "toward", "matching", "container"])
            text_parts.append("push a colored block toward its matching container")
        if info.get("object_delivered"):
            tokens.extend(["delivered", "container", "match", "reward"])
            text_parts.append("block delivered into the correct color container")
        if info.get("collision"):
            tokens.extend(["blocked", "wall", "object"])
            text_parts.append("blocked by wall or object")
        tokens.extend(["remaining", _num(info.get("remaining_objects", 0), digits=0)])
    elif game_id == "nibbles":
        tokens.extend(["snake", "head", "food"])
        if info.get("collected"):
            tokens.extend(["eat", "green", "yellow", "blue", "food", "reward"])
            text_parts.append("eat bright reward food")
        if info.get("penalty_food"):
            tokens.extend(["avoid", "red", "poison", "negative"])
            text_parts.append("avoid red poison food")
        if info.get("collision"):
            tokens.extend(["wall", "body", "crash"])
            text_parts.append("snake collision")
        tokens.extend(["score", _num(info.get("score", 0.0)), "target", _num(info.get("target_score", 0.0))])
    elif game_id == "angry_blocks":
        tokens.extend(["slingshot", "projectile", "gravity", "tower"])
        if info.get("launch"):
            tokens.extend(["launch", "arc", "force", "power", _num(info.get("power", 0.0))])
            text_parts.append("launch a square projectile in a gravity arc")
        if info.get("hits"):
            tokens.extend(["hit", "block", "impact", "falling"])
            text_parts.append("projectile hits blocks")
        if info.get("target_hits"):
            tokens.extend(["green", "target", "reward"])
            text_parts.append("target block hit")
        tokens.extend(["shots", _num(info.get("shots_left", 0), digits=0)])
    elif game_id == "platformer":
        tokens.extend(["player", "platform", "ground", "gravity"])
        if info.get("jumped"):
            tokens.extend(["jump", "upward", "airborne"])
            text_parts.append("jump upward under gravity")
        if info.get("collected"):
            tokens.extend(["collect", "bright", "reward", "pickup"])
            text_parts.append("collect bright pickup")
        if info.get("hazard"):
            tokens.extend(["red", "hazard", "trap", "negative"])
            text_parts.append("red hazard causes penalty")
        if float(info.get("progress", 0.0) or 0.0) > 0:
            tokens.extend(["progress", "right", "goal"])
            text_parts.append("move right toward goal")
    elif game_id == "phoenix_duel":
        tokens.extend(["ship", "phoenix", "enemy", "bullet", "hand-eye", "aim"])
        if info.get("fired"):
            tokens.extend(["fire", "laser", "upward", "shot"])
            text_parts.append("fire a laser at moving phoenix targets")
        if info.get("hits"):
            tokens.extend(["hit", "enemy", "reward", "sync"])
            text_parts.append("laser hits a moving enemy")
        if info.get("hazard"):
            tokens.extend(["avoid", "red", "enemy", "bullet", "negative"])
            text_parts.append("enemy bullet hits the ship")
        if info.get("shielded"):
            tokens.extend(["shield", "defense"])
            text_parts.append("shield blocks danger")
        tokens.extend(["score", _num(info.get("score", 0.0)), "health", _num(info.get("health", 0), digits=0)])
    elif game_id == "sky_catch":
        tokens.extend(["basket", "falling", "object", "sky", "hand-eye", "timing"])
        if info.get("collected"):
            tokens.extend(["catch", "green", "yellow", "reward", "falling", "tone"])
            text_parts.append("catch bright falling reward objects")
        if info.get("neutral_collected"):
            tokens.extend(["catch", "cyan", "neutral"])
            text_parts.append("catch neutral falling object")
        if info.get("penalty_food"):
            tokens.extend(["avoid", "red", "penalty", "low", "sound"])
            text_parts.append("red falling object gives penalty")
        if info.get("hazard"):
            tokens.extend(["avoid", "purple", "danger", "deadly", "negative"])
            text_parts.append("deadly falling object hurts the player")
        fall_sound = str(info.get("fall_sound", ""))
        if fall_sound:
            tokens.extend(["fall-sound", fall_sound])
        tokens.extend(["score", _num(info.get("score", 0.0)), "health", _num(info.get("health", 0), digits=0)])
    elif game_id == "dr_capsule":
        tokens.extend(["capsule", "bottle", "stack", "rotate", "gravity", "color", "match-four", "virus"])
        if info.get("capsule_held"):
            tokens.extend(["hold", "swap", "reserve", "capsule"])
            text_parts.append("hold and swap the falling capsule")
        if info.get("hard_drop"):
            tokens.extend(["hard", "drop", "fast", "fall"])
            text_parts.append("hard drop capsule to the stack")
        if info.get("capsule_moved"):
            tokens.extend(["move", "left", "right", "align"])
            text_parts.append("move falling capsule inside the bottle")
        if info.get("capsule_rotated"):
            tokens.extend(["rotate", "bicolor", "capsule"])
            text_parts.append("rotate a bicolor capsule")
        if info.get("same_color_pairs"):
            tokens.extend(["same", "color", "pair", "small", "reward"])
            text_parts.append("stack same-color halves for small reward")
        if info.get("pieces_cleared"):
            tokens.extend(["clear", "four", "same", "color", "spike", "reward", "falling", "halves"])
            text_parts.append("four same-color pieces disappear and halves fall")
        if info.get("viruses_cleared"):
            tokens.extend(["virus", "red", "circle", "cleared", "reward"])
            text_parts.append("red virus circles cleared from the bottle")
        if info.get("mismatch_pairs"):
            tokens.extend(["mismatch", "different", "color", "penalty"])
            text_parts.append("different-color halves touch and give penalty")
        if info.get("hazard"):
            tokens.extend(["overflow", "top", "bottle", "lose", "negative"])
            text_parts.append("bottle overflows at the top")
        if info.get("goal_reached"):
            tokens.extend(["all", "viruses", "cleared", "win"])
            text_parts.append("all red viruses cleared from the bottle")
        tokens.extend(
            [
                "cleared",
                _num(info.get("pieces_cleared", 0), digits=0),
                "viruses-left",
                _num(info.get("viruses_remaining", 0), digits=0),
                "combo",
                _num(info.get("combo", 0), digits=0),
            ]
        )
    elif game_id == "fabulous_fred":
        tokens.extend(["memory", "sequence", "color", "sound", "turn", "simon", "fabulous-fred"])
        turn = str(info.get("turn", "machine"))
        tokens.extend(["turn", turn])
        if turn == "machine":
            text_parts.append("watch the machine's color and sound sequence")
        elif turn == "player":
            text_parts.append("repeat the sequence in the same order")
            
        if info.get("correct_touch"):
            tokens.extend(["correct", "match", "reward"])
            text_parts.append("correctly touch the matching square")
        if info.get("wrong_touch"):
            tokens.extend(["wrong", "error", "penalty"])
            text_parts.append("wrong square touched, sequence failed")
        if info.get("timeout"):
            tokens.extend(["timeout", "slow", "penalty"])
            text_parts.append("time ran out, failed to act")
        if info.get("round_completed"):
            tokens.extend(["round", "completed", "bonus", "faster"])
            text_parts.append("round completed, next sequence is longer and faster")
            
        tokens.extend(
            [
                "sequence-len",
                _num(info.get("sequence_len", 3), digits=0),
                "player-index",
                _num(info.get("player_seq_index", 0), digits=0),
                "timer",
                _num(info.get("player_timer", 200), digits=0),
            ]
        )
    else:
        tokens.extend(["arcade", "observe", "act"])

    train = float(reward.get("train", 0.0) or 0.0)
    if train > 0.2:
        tokens.extend(["positive", "reward"])
    elif train < -0.05:
        tokens.extend(["negative", "penalty"])
    else:
        tokens.append("neutral")

    ax = info.get("agent_x")
    ay = info.get("agent_y")
    tx = info.get("target_x")
    ty = info.get("target_y")
    if ax is not None and ay is not None:
        tokens.extend(["agent-x", _num(ax, digits=0), "agent-y", _num(ay, digits=0)])
    if tx is not None and ty is not None:
        tokens.extend(["target-x", _num(tx, digits=0), "target-y", _num(ty, digits=0)])

    clean_tokens: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        tok = str(tok).strip().lower()
        if not tok:
            continue
        if tok in seen:
            continue
        clean_tokens.append(tok)
        seen.add(tok)
    text = "; ".join(text_parts) if text_parts else "observe action effect"
    token_budget = max(1, int(max_tokens or 48))
    prompt_tokens = clean_tokens[:token_budget]
    return {
        "language_text": text,
        "language_tokens": clean_tokens,
        "language_prompt": " ".join(prompt_tokens),
        "language_token_budget": token_budget,
        "language_prompt_tokens": len(prompt_tokens),
        "action_key": str(action_key),
        "action_data": dict(action_data or {}),
    }
