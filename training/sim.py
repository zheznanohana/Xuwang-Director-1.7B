"""Abstract simulator of the final demon-king fight (battle 11), run without Godot.

Numbers mirror scripts/combat/*.gd; every simplification is tagged APPROX so it can
be recalibrated once real in-game decision logs exist.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

W, H = 8, 6                      # 8x6 board
PLAYER_MOVE = 3                  # battle_core.gd:57 MOVE_RANGE
SKILL_RANGE = 4                  # battle_turns.gd:236-241 "distance <= 4"
SPELL_RANGE = 4                  # APPROX: typical player card reach
PLAYER_DMG_IN_RANGE = 12         # APPROX design target: ~6-7 turn fight. UNCALIBRATED: no real logs exist for this boss yet
PLAYER_DMG_MELEE_BONUS = 3       # APPROX
PLAYER_DMG_OUT_OF_RANGE = 3      # APPROX: buffs / chip damage only
PLAYER_BLOCK_GAIN = 6            # APPROX
HAZARD_DAMAGE = 2                # APPROX: ignite/charge terrain (burn/shock statuses collapsed to flat damage)
HAZARD_TURNS = 2                 # battle_turns.gd: _ignite(c, 2) / _charge(c, 2)

BOSS_BASE = {"hp": 64, "atk": 5, "move": 2, "block": 6, "cell": (6, 2)}  # journey_core.gd:184
PHASE_BLOCK = 6                  # battle_turns.gd:93
PLAYER_DEFAULT_MAX_HP = 34       # battle_setup.gd:722

SKILL_POOL = [                   # battle_turns.gd:8 DEMON_KING_SKILL_CYCLE
    "demon_soul_chain", "demon_abyss_cross", "demon_crown_nova",
    "demon_eclipse_diagonal", "demon_world_rend",
]
POOL_INTENT = {
    "demon_soul_chain": "siphon", "demon_abyss_cross": "abyss_cross",
    "demon_crown_nova": "nova", "demon_eclipse_diagonal": "eclipse_diagonal",
    "demon_world_rend": "world_rend",
}
# Decision vocabulary of the learned policy. Starfall/cataclysm stay owned by the upper FSM.
OPTIONS = ["siphon", "abyss_cross", "nova", "eclipse_diagonal", "world_rend", "advance", "move"]
DAMAGE_DELTA = {                 # battle_turns.gd execution branches
    "siphon": -2, "abyss_cross": -1, "nova": 0, "eclipse_diagonal": -1,
    "world_rend": 0, "advance": 0, "cataclysm": 1,
}
RANGE_LIMITED = {"siphon", "nova", "world_rend"}
TERRAIN = {"abyss_cross", "nova", "eclipse_diagonal", "world_rend", "cataclysm"}
DAMAGE_INTENTS = set(DAMAGE_DELTA)
ARENAS = {                       # nemesis_planner.gd ARENAS
    "none": [],
    "open_crown": [(2, 1), (2, 5), (5, 0), (5, 4)],
    "split_forge": [(3, 1), (3, 4), (5, 2)],
    "narrow_ring": [(1, 0), (1, 5), (4, 0), (4, 5), (6, 3)],
}


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_board(c):
    return 0 <= c[0] < W and 0 <= c[1] < H


def square(c, r):
    return [(x, y) for x in range(c[0] - r, c[0] + r + 1)
            for y in range(c[1] - r, c[1] + r + 1) if in_board((x, y))]


def row(c):
    return [(x, c[1]) for x in range(W)]


def cross(c):
    return row(c) + [(c[0], y) for y in range(H) if y != c[1]]


def diagonals(c):
    return [(x, y) for x in range(W) for y in range(H) if abs(x - c[0]) == abs(y - c[1])]


@dataclass
class State:
    arena: str
    blocked: frozenset
    boss_cell: tuple
    boss_hp: int
    boss_max: int
    boss_block: int
    boss_atk: int
    boss_move: int
    player_cell: tuple
    player_hp: int
    player_max: int
    player_hp_start: int = 0
    player_block: int = 0
    player_chill: int = 0
    turn: int = 0
    skill_offset: int = 0
    threat_debt: int = 0
    phase2: bool = False
    starfall_left: int = 0
    starfall_used: bool = False
    hazards: dict = field(default_factory=dict)
    intent: str = ""
    locked_cells: frozenset = frozenset()
    locked_path: list = field(default_factory=list)
    boss_hist: list = field(default_factory=list)     # executed boss intents
    player_hist: list = field(default_factory=list)   # public per-turn player records

    @property
    def distance(self):
        return manhattan(self.boss_cell, self.player_cell)


def new_state(rng, arena=None, player_max=None, player_hp_frac=None, skill_offset=None):
    arena = arena or rng.choice(list(ARENAS))
    pmax = player_max or rng.randint(30, 46)
    frac = player_hp_frac if player_hp_frac is not None else rng.uniform(0.55, 1.0)
    return State(
        arena=arena, blocked=frozenset(ARENAS[arena]),
        boss_cell=BOSS_BASE["cell"], boss_hp=BOSS_BASE["hp"], boss_max=BOSS_BASE["hp"],
        boss_block=BOSS_BASE["block"], boss_atk=BOSS_BASE["atk"], boss_move=BOSS_BASE["move"],
        player_cell=(1, 2), player_hp=max(1, round(pmax * frac)), player_max=pmax,
        skill_offset=rng.randrange(5) if skill_offset is None else skill_offset,
        player_hp_start=max(1, round(pmax * frac)),
    )


def reachable(s, start, steps, avoid):
    """BFS over passable cells; returns {cell: steps}. `avoid` is the other unit's cell."""
    seen = {start: 0}
    queue = deque([start])
    while queue:
        c = queue.popleft()
        if seen[c] == steps:
            continue
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + d[0], c[1] + d[1])
            if in_board(n) and n not in s.blocked and n != avoid and n not in seen:
                seen[n] = seen[c] + 1
                queue.append(n)
    return seen


def boss_path_toward(s, target, steps):
    """Shortest path cells (excluding start) the boss would walk toward `target`."""
    if steps <= 0 or manhattan(s.boss_cell, target) <= 1:
        return []
    prev = {s.boss_cell: None}
    queue = deque([s.boss_cell])
    goal = None
    while queue:
        c = queue.popleft()
        if manhattan(c, target) <= 1 and c != s.boss_cell:
            goal = c
            break
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + d[0], c[1] + d[1])
            if in_board(n) and n not in s.blocked and n != target and n not in prev:
                prev[n] = c
                queue.append(n)
    if goal is None:
        return []
    path = []
    while goal != s.boss_cell:
        path.append(goal)
        goal = prev[goal]
    path.reverse()
    return path[:steps]


def intent_cells(s, intent, target):
    if intent == "siphon":
        return [target]
    if intent == "nova":
        return square(target, 1)
    if intent == "world_rend":
        return row(target)
    if intent == "abyss_cross":
        return cross(target)
    if intent == "eclipse_diagonal":
        return diagonals(target)
    if intent in ("cataclysm", "starfall_warning"):
        return square(target, 2)
    if intent == "advance":
        # APPROX of _locked_path_cells: walked path plus the locked target cell.
        return boss_path_toward(s, target, s.boss_move) + [target]
    return []


def intent_damage(s, intent):
    return max(1, s.boss_atk + DAMAGE_DELTA[intent]) if intent in DAMAGE_DELTA else 0


def legal_options(s):
    legal = []
    for opt in OPTIONS:
        if opt in RANGE_LIMITED and s.distance > SKILL_RANGE:
            continue
        if opt == "move" and s.threat_debt >= 1:
            continue  # threat-debt contract: a pure non-damage turn cannot follow another
        legal.append(opt)
    return legal


def scripted_intent(s):
    """Upper-layer FSM: phase shift + starfall countdown. Returns an intent or None."""
    if not s.phase2 and s.boss_hp * 2 <= s.boss_max:
        s.phase2 = True
        s.boss_block += PHASE_BLOCK
        if not s.starfall_used:
            s.starfall_left = 3
            s.starfall_used = True
    if s.starfall_left > 0:
        return "cataclysm" if s.starfall_left == 1 else "starfall_warning"
    return None


def fsm_choice(s):
    """Replica of the current hand-written policy (battle_turns.gd _base_enemy_intent path)."""
    skill = SKILL_POOL[(s.turn - 1 + s.skill_offset) % len(SKILL_POOL)]
    intent = POOL_INTENT[skill]
    if intent in RANGE_LIMITED and s.distance > SKILL_RANGE:
        intent = "move"
    if s.threat_debt >= 1 and intent == "move":
        # _enemy_forced_pressure_intent; adjacent "attack" behaves like a zero-step advance.
        if s.distance == 1:
            intent = "advance"
        elif s.distance <= SKILL_RANGE:
            intent = POOL_INTENT[skill]
        else:
            intent = "advance"
    return intent


def telegraph(s, intent):
    s.intent = intent
    cells = intent_cells(s, intent, s.player_cell)
    s.locked_cells = frozenset(cells)
    s.locked_path = cells[:-1] if intent == "advance" else []


def _boss_walk(s, path):
    for c in path:
        if c == s.player_cell:
            break
        s.boss_cell = c


def resolve_turn(s, dest, blocking):
    """Player moves to `dest` (already validated) and casts; then the boss executes the locked intent."""
    events = {"hit": False, "damage_taken": 0, "damage_dealt": 0}
    start = s.player_cell
    was_in_area = start in s.locked_cells
    s.player_cell = dest
    s.player_chill = max(0, s.player_chill - 1)
    if dest in s.hazards:
        s.player_hp -= HAZARD_DAMAGE
        events["damage_taken"] += HAZARD_DAMAGE
    dist = manhattan(dest, s.boss_cell)
    dealt = PLAYER_DMG_IN_RANGE if dist <= SPELL_RANGE else PLAYER_DMG_OUT_OF_RANGE
    if dist <= 1:
        dealt += PLAYER_DMG_MELEE_BONUS
    s.player_block = 0
    if blocking:
        s.player_block = PLAYER_BLOCK_GAIN
        dealt //= 2
    absorbed = min(s.boss_block, dealt)
    s.boss_block -= absorbed
    s.boss_hp -= dealt - absorbed
    events["damage_dealt"] = dealt
    s.player_hist.append({
        "from": start, "to": dest, "dist_after": dist, "was_in_area": was_in_area,
        "stayed_in_area": dest in s.locked_cells, "blocked": blocking, "intent": s.intent,
    })
    if s.boss_hp <= 0 or s.player_hp <= 0:
        return events

    intent = s.intent
    hit = s.player_cell in s.locked_cells and intent in DAMAGE_INTENTS
    if intent == "advance":
        # _advance_along_locked_path: walk the frozen path; contact only if the player stayed on it.
        _boss_walk(s, s.locked_path)
        hit = hit and manhattan(s.boss_cell, s.player_cell) <= 1
    if hit:
        dmg = intent_damage(s, intent)
        blocked = min(s.player_block, dmg)
        s.player_hp -= dmg - blocked
        events["hit"] = True
        events["damage_taken"] += dmg - blocked
        if intent == "siphon":
            s.player_chill += 1
            s.boss_block += 2
    if intent in TERRAIN:
        for c in s.locked_cells:
            if c != s.boss_cell:
                s.hazards[c] = HAZARD_TURNS + 1   # decays once below before the player can act
    if intent == "cataclysm":
        s.starfall_left = 0
    elif intent == "starfall_warning":
        s.starfall_left = max(s.starfall_left - 1, 1)
        _boss_walk(s, boss_path_toward(s, s.player_cell, s.boss_move))
    elif intent == "move":
        _boss_walk(s, boss_path_toward(s, s.player_cell, s.boss_move))
    s.threat_debt = 1 if intent == "move" else 0
    s.hazards = {c: t - 1 for c, t in s.hazards.items() if t > 1}
    s.boss_hist.append(intent)
    s.player_hist[-1]["hit"] = events["hit"]
    return events


def done(s):
    if s.boss_hp <= 0:
        return "player_win"
    if s.player_hp <= 0:
        return "boss_win"
    if s.turn >= 30:
        return "timeout"
    return None
