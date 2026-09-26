"""Player bots and boss teachers for the decision simulator.

Boss teacher follows piKL (Jacob et al., ICML 2022): pi(a) ∝ anchor(a) * exp(Q(a) / tau).
The anchor is the shipped FSM (designer intent); Q is a one-turn lookahead against a
player model fitted only from public history. tau trades "designer rhythm" for "adaptive".
Q scores fight quality, not boss win rate: it rewards forcing the player to trade range for
safety, penalises self-repetition and eases off when the player is nearly dead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import sim

HABITS = {"none": (0.0, 0.0), "left": (-1.0, 0.0), "right": (1.0, 0.0),
          "up": (0.0, -1.0), "down": (0.0, 1.0), "corner": None}


@dataclass(frozen=True)
class PlayerParams:
    d_pref: int = 3          # preferred distance to the boss
    habit: str = "none"      # escape direction habit (public, learnable by the boss)
    greed: float = 0.2       # 0 = always dodges, 1 = ignores telegraphs
    block_rate: float = 0.0  # share of turns spent turtling
    noise: float = 0.4       # softmax temperature over destinations
    skill: float = 1.0       # scales telegraph misread probability (lower = sharper reader)

    def as_dict(self):
        return asdict(self)


def _softmax_pick(items, scores, temp, rng):
    m = max(scores)
    weights = [math.exp((x - m) / max(temp, 1e-6)) for x in scores]
    r = rng.random() * sum(weights)
    for item, w in zip(items, weights):
        r -= w
        if r <= 0:
            return item
    return items[-1]


def _habit_score(cell, origin, habit_vec, corner):
    if corner:
        corners = ((0, 0), (sim.W - 1, 0), (0, sim.H - 1), (sim.W - 1, sim.H - 1))
        return -min(sim.manhattan(cell, c) for c in corners) / 4.0
    return ((cell[0] - origin[0]) * habit_vec[0] + (cell[1] - origin[1]) * habit_vec[1]) / 3.0


DAMAGE_VALUE = 0.6  # APPROX: HP a player will trade for one point of boss damage


def expected_dealt(dist):
    dealt = sim.PLAYER_DMG_IN_RANGE if dist <= sim.SPELL_RANGE else sim.PLAYER_DMG_OUT_OF_RANGE
    return dealt + (sim.PLAYER_DMG_MELEE_BONUS if dist <= 1 else 0)


def destination_utility(s, cell, d_pref, habit_vec, corner, greed):
    """HP-equivalent value of ending the move on `cell`: dodge vs keep dealing damage."""
    danger = sim.intent_damage(s, s.intent) if cell in s.locked_cells and s.intent in sim.DAMAGE_INTENTS else 0
    dist = sim.manhattan(cell, s.boss_cell)
    return (-danger * (1.0 - greed)
            - sim.HAZARD_DAMAGE * (cell in s.hazards)
            - 0.3 * abs(dist - d_pref)
            + 1.0 * _habit_score(cell, s.player_cell, habit_vec, corner)
            + DAMAGE_VALUE * expected_dealt(dist))


# APPROX human telegraph-reading error. Geometry alone always leaves a safe in-range cell on
# this board, so real hits come from misreads; validate with in-game "hit rate by intent x recency".
READ_ERROR = {"siphon": 0.10, "nova": 0.10, "world_rend": 0.12, "abyss_cross": 0.18,
              "eclipse_diagonal": 0.22, "advance": 0.20, "cataclysm": 0.15, "starfall_warning": 0.0, "move": 0.0}
READ_ERROR_SCALE = 2.0


def read_error(s, intent, skill=1.0):
    e = READ_ERROR.get(intent, 0.0) * skill * READ_ERROR_SCALE
    if intent in s.boss_hist[-3:]:
        e *= 0.6   # seen recently
    if s.turn > 5 and intent == sim.fsm_choice(s):
        e *= 0.6   # matches the memorised rotation
    return min(e, 0.9)


def player_move_range(s):
    return max(1, sim.PLAYER_MOVE - s.player_chill)


def player_act(s, p: PlayerParams, rng):
    cells = list(sim.reachable(s, s.player_cell, player_move_range(s), s.boss_cell))
    corner = p.habit == "corner"
    vec = HABITS[p.habit] or (0.0, 0.0)
    greed = 1.0 if rng.random() < read_error(s, s.intent, p.skill) else p.greed  # misread = no dodge
    scores = [destination_utility(s, c, p.d_pref, vec, corner, greed) for c in cells]
    dest = _softmax_pick(cells, scores, p.noise, rng)
    return dest, rng.random() < p.block_rate


# ---------------------------------------------------------------- boss side

def fitted_player_model(s):
    """Player model from public history only (no hand, no hidden params)."""
    recent = s.player_hist[-3:]
    if not recent:
        return 3.0, (0.0, 0.0), 0.2
    d_pref = sum(h["dist_after"] for h in recent) / len(recent)
    dx = sum(h["to"][0] - h["from"][0] for h in recent) / (3.0 * len(recent))
    dy = sum(h["to"][1] - h["from"][1] for h in recent) / (3.0 * len(recent))
    dodges = [h for h in s.player_hist[-4:] if h["was_in_area"]]
    greed = (sum(h["stayed_in_area"] for h in dodges) + 0.2) / (len(dodges) + 1.0)
    return d_pref, (dx, dy), greed


def fsm_policy(s):
    choice = sim.fsm_choice(s)
    return {a: (1.0 if a == choice else 0.0) for a in sim.legal_options(s)}, {"anchor": choice}


def lookahead_q(s, option, model):
    d_pref, vec, greed = model
    saved = (s.intent, s.locked_cells)
    s.intent = option
    s.locked_cells = frozenset(sim.intent_cells(s, option, s.player_cell))
    dests = list(sim.reachable(s, s.player_cell, player_move_range(s), s.boss_cell))
    err = read_error(s, option)
    p_hit = p_denied = 0.0
    for weight, g in ((1.0 - err, greed), (err, 1.0)):   # read correctly / misread
        if weight <= 0:
            continue
        scores = [destination_utility(s, c, d_pref, vec, False, g) for c in dests]
        m = max(scores)
        ws = [math.exp((x - m) / 0.5) for x in scores]
        z = sum(ws)
        p_hit += weight * sum(w for c, w in zip(dests, ws) if c in s.locked_cells) / z
        p_denied += weight * sum(w for c, w in zip(dests, ws) if sim.manhattan(c, s.boss_cell) > sim.SPELL_RANGE) / z
    s.intent, s.locked_cells = saved
    if option == "advance":
        p_hit *= 0.5  # contact also needs the frozen path to end adjacent
    dmg = sim.intent_damage(s, option) if option in sim.DAMAGE_INTENTS else 0
    mercy = 0.4 if s.player_hp <= 0.3 * s.player_max else 1.0
    closing = 1.0 if option in ("advance", "move") and s.distance > sim.SKILL_RANGE else 0.0
    repeat = sum(1 for x in s.boss_hist[-2:] if x == option)
    q = p_hit * dmg * mercy + 0.35 * p_denied * sim.PLAYER_DMG_IN_RANGE + closing - 1.2 * repeat
    return q, p_hit, p_denied


def pikl_policy(s, tau=1.0, beta=0.5):
    legal = sim.legal_options(s)
    anchor_choice = sim.fsm_choice(s)
    model = fitted_player_model(s)
    qs, hits, denied, logits = {}, {}, {}, {}
    for a in legal:
        q, h, d = lookahead_q(s, a, model)
        share = (1 - beta) / len(legal)
        anchor = beta + share if a == anchor_choice else share
        qs[a], hits[a], denied[a] = round(q, 4), round(h, 4), round(d, 4)
        logits[a] = math.log(anchor) + q / tau
    m = max(logits.values())
    exp = {a: math.exp(v - m) for a, v in logits.items()}
    z = sum(exp.values())
    dist = {a: v / z for a, v in exp.items()}
    info = {"q": qs, "p_hit": hits, "p_denied": denied, "anchor": anchor_choice,
            "model": {"d_pref": round(model[0], 3), "habit": [round(model[1][0], 3), round(model[1][1], 3)],
                      "greed": round(model[2], 3)},
            "tau": tau, "beta": beta}
    return dist, info


def sample(dist, rng):
    r = rng.random()
    for a, p in dist.items():
        r -= p
        if r <= 0:
            return a
    return next(iter(dist))
