"""Public state -> record fields, fixed feature vector, and LM prompt.

Everything here must stay reproducible in GDScript from information the player can see:
positions, HP, block, telegraph history and the player's past moves. Never the hand/deck.
"""
from __future__ import annotations

import sim

SCHEMA = "boss_tactic@1"
RENDERER_VERSION = 1
LABEL_WORDS = {  # candidate verbalizer words, first single-token one wins (checked in Colab)
    "siphon": [" siphon", " drain", " chain"],
    "abyss_cross": [" cross", " plus"],
    "nova": [" nova", " burst", " blast"],
    "eclipse_diagonal": [" diagonal", " eclipse", " slant"],
    "world_rend": [" row", " rend", " line"],
    "advance": [" advance", " press", " charge"],
    "move": [" move", " walk", " reposition"],
}
DIRS = {(0, 0): "stay", (1, 0): "E", (-1, 0): "W", (0, 1): "S", (0, -1): "N",
        (1, 1): "SE", (1, -1): "NE", (-1, 1): "SW", (-1, -1): "NW"}
DIR_NAMES = list(dict.fromkeys(DIRS.values()))


def _sign(v):
    return (v > 0) - (v < 0)


def move_dir(h):
    return DIRS[(_sign(h["to"][0] - h["from"][0]), _sign(h["to"][1] - h["from"][1]))]


def option_features(s):
    """Per-option tactical facts the engine can precompute (the model decides, it does not measure)."""
    from agents import player_move_range
    dests = sim.reachable(s, s.player_cell, player_move_range(s), s.boss_cell)
    legal = set(sim.legal_options(s))
    out = {}
    for opt in sim.OPTIONS:
        cells = set(sim.intent_cells(s, opt, s.player_cell)) if opt in legal else set()
        safe = [c for c in dests if c not in cells]
        out[opt] = {
            "legal": opt in legal,
            "dmg": sim.intent_damage(s, opt) if opt in sim.DAMAGE_INTENTS else 0,
            "area": len(cells),
            "safe_cells": len(safe),
            "safe_in_range": sum(1 for c in safe if sim.manhattan(c, s.boss_cell) <= sim.SPELL_RANGE),
            "used_last2": sum(1 for x in s.boss_hist[-2:] if x == opt),
        }
    return out


def public_state(s):
    last = s.player_hist[-4:]
    dodge_turns = [h for h in s.player_hist[-4:] if h["was_in_area"]]
    return {
        "arena": s.arena,
        "turn": s.turn,
        "phase": 2 if s.phase2 else 1,
        "boss_hp_pct": round(100 * s.boss_hp / s.boss_max),
        "boss_block": s.boss_block,
        "boss_cell": list(s.boss_cell),
        "player_hp_pct": round(100 * s.player_hp / s.player_max),
        "player_block": s.player_block,
        "player_chill": s.player_chill,
        "player_cell": list(s.player_cell),
        "dist": s.distance,
        "threat_debt": s.threat_debt,
        "player_last_moves": [move_dir(h) for h in last],
        "player_last_dists": [h["dist_after"] for h in last],
        "player_hits_last3": sum(1 for h in s.player_hist[-3:] if h.get("hit")),
        "player_stayed_rate": round(sum(h["stayed_in_area"] for h in dodge_turns) / len(dodge_turns), 2) if dodge_turns else None,
        "player_block_rate": round(sum(h["blocked"] for h in last) / len(last), 2) if last else None,
        "boss_last3": s.boss_hist[-3:],
        "fsm_suggestion": sim.fsm_choice(s),
    }


def vectorize(state, opts):
    """Fixed-length float vector for the MLP track (mirror this exactly in GDScript)."""
    v = [
        state["phase"] - 1, state["boss_hp_pct"] / 100, state["boss_block"] / 20,
        state["player_hp_pct"] / 100, state["player_block"] / 10, state["player_chill"] / 2,
        state["boss_cell"][0] / 7, state["boss_cell"][1] / 5,
        state["player_cell"][0] / 7, state["player_cell"][1] / 5,
        state["dist"] / 12, state["threat_debt"], min(state["turn"], 15) / 15,
        state["player_hits_last3"] / 3,
        state["player_stayed_rate"] if state["player_stayed_rate"] is not None else 0.2,
        state["player_block_rate"] if state["player_block_rate"] is not None else 0.0,
        len(state["player_last_moves"]) / 4,
    ]
    moves = state["player_last_moves"]
    for name in DIR_NAMES:  # habit histogram over the last 4 moves
        v.append(sum(1 for m in moves if m == name) / 4)
    dists = state["player_last_dists"]
    v.append(sum(dists) / len(dists) / 12 if dists else 0.25)
    for opt in sim.OPTIONS:
        o = opts[opt]
        v += [float(o["legal"]), o["dmg"] / 6, o["area"] / 20, o["safe_cells"] / 25,
              o["safe_in_range"] / 25, o["used_last2"] / 2, float(state["fsm_suggestion"] == opt)]
    return v


def render_prompt(state, opts):
    """Compact English prompt; ends right before the single answer token."""
    legal = [o for o in sim.OPTIONS if opts[o]["legal"]]
    lines = [
        f"<decision> {SCHEMA}",
        "<options> " + " ".join(LABEL_WORDS[o][0].strip() for o in legal),
        "<state>",
        f"phase={state['phase']} turn={state['turn']} arena={state['arena']}",
        f"boss_hp={state['boss_hp_pct']} boss_block={state['boss_block']} boss_at={state['boss_cell'][0]},{state['boss_cell'][1]}",
        f"player_hp={state['player_hp_pct']} player_block={state['player_block']} chill={state['player_chill']} player_at={state['player_cell'][0]},{state['player_cell'][1]}",
        f"dist={state['dist']} debt={state['threat_debt']} rhythm={LABEL_WORDS[state['fsm_suggestion']][0].strip()}",
        "<history>",
        "player_moves=" + (" ".join(state["player_last_moves"]) or "none"),
        "player_dists=" + (" ".join(map(str, state["player_last_dists"])) or "none"),
        f"player_hits={state['player_hits_last3']} stayed_rate={state['player_stayed_rate']} block_rate={state['player_block_rate']}",
        "boss_last=" + (" ".join(LABEL_WORDS[x][0].strip() if x in LABEL_WORDS else x for x in state["boss_last3"]) or "none"),
        "<tactics>",
    ]
    for o in legal:
        f = opts[o]
        lines.append(f"{LABEL_WORDS[o][0].strip()}: dmg={f['dmg']} area={f['area']} safe={f['safe_cells']} safe_in_range={f['safe_in_range']} recent={f['used_last2']}")
    lines.append("<answer>")
    return "\n".join(lines)
