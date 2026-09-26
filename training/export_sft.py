"""Merge labelled channel files into one multi-task training file.

  python export_sft.py --out out/sft_v1.jsonl

Each output row: {channel, split, prompt, targets: {field: {kind, options, dist, legal?}}}
  kind "choice": soft distribution over a fixed option list (legal mask optional)
  kind "multi" : independent per-tag probabilities (engine takes top-k / threshold)
Prompts are compact English key=value text; the model never sees teacher-only fields
(latent skill, bot parameters, rule-teacher internals).
"""
from __future__ import annotations

import argparse
import math
import glob
import json
import re

import channels as ch
import features
import sim

_CARD_FX = re.compile(r"^c\d+_fx(\d+)$")

FIELD_SPECS = {
    "boss_tactic@1": {"tactic": ("choice", sim.OPTIONS)},
    "difficulty_adaptation@1": {
        "challenge_delta": ("choice", list(range(-2, 6))),
        "hp_bucket": ("choice", list(range(-5, 16))),
        "attack_delta": ("choice", list(range(-1, 3))),
        "block_delta": ("choice", list(range(-2, 7))),
        "move_delta": ("choice", list(range(-1, 3))),
        "trait_id": ("choice", ch.ADAPT_TRAITS),
        "reason_tags": ("multi", ch.ADAPT_REASONS),
    },
    "nemesis_director@1": {
        "archetype": ("choice", ch.NEMESIS_ARCHETYPES),
        "phase_tag": ("choice", ch.NEMESIS_PHASES),
        "arena_tag": ("choice", ch.NEMESIS_ARENAS),
        "ultimate_card": ("choice", ch.NEMESIS_ULTIMATES),
        "player_tags": ("multi", ch.NEMESIS_PLAYER_TAGS),
        "mechanic_tags": ("multi", ch.NEMESIS_MECHANICS),
    },
    "fate_analysis@1": {**{f"adj_{r}": ("choice", [v for v, _ in ch.FATE_ADJ]) for r in ch.FATE_ROUTES},
                        "bias_shift": ("choice", [v for v, _ in ch.FATE_BIAS_SHIFT]),
                        "uncertainty": ("choice", [v for v, _ in ch.FATE_UNCERTAINTY])},
    "music_scene@1": {"style_id": ("choice", list(ch.MUSIC_STYLES)),
                      "density_profile": ("choice", list(ch.MUSIC_DENSITY)),
                      "rhythm_profile": ("choice", list(ch.MUSIC_RHYTHM)),
                      "bpm_shift": ("choice", [v for v, _ in ch.MUSIC_BPM]),
                      "tension_shift": ("choice", [v for v, _ in ch.MUSIC_TENSION])},
}


def _card_field_specs():
    spec = {}
    for i in range(ch.CARD_MAX_CARDS):
        spec[f"c{i}_target"] = ("choice", list(ch.CARD_TARGETS))
        spec[f"c{i}_cost"] = ("choice", list(ch.CARD_COSTS))
        spec[f"c{i}_range"] = ("choice", list(ch.CARD_RANGES))
        for j in range(ch.CARD_MAX_EFFECTS):
            spec[f"c{i}_fx{j}"] = ("choice", ch.CARD_ALL_EFFECTS + ["none"])
            spec[f"c{i}_fxv{j}"] = ("choice", list(ch.CARD_VALUES))
            spec[f"c{i}_fxc{j}"] = ("choice", list(ch.CARD_CONDITIONS))
        spec[f"c{i}_tags"] = ("multi", list(ch.CARD_BUILD_TAGS))
        spec[f"c{i}_keywords"] = ("multi", list(ch.CARD_KEYWORDS))
    return spec


FIELD_SPECS["card_procurement@1"] = _card_field_specs()
FIELD_SPECS["music_score@1"] = {
    **{f"prog{b}": ("choice", list(ch.MUSIC_PROG)) for b in range(ch.MUSIC_BARS)},
    **{f"s{i}": ("choice", list(ch.MUSIC_STEP_TOKENS)) for i in range(ch.MUSIC_BARS * ch.MUSIC_STEPS_PER_BAR)},
}
FIELD_SPECS["camp_assessment@1"] = {
    "tempo": ("choice", list(ch.CAMP_TEMPO)),
    "damage": ("choice", list(ch.CAMP_DAMAGE)),
    "tactics": ("choice", list(ch.CAMP_TACTICS)),
    "overall": ("choice", list(ch.CAMP_OVERALL)),
}
FIELD_SPECS["encounter_flavor@1"] = {
    "tone": ("choice", list(ch.FLAVOR_TONES)),
    "style_tags": ("multi", list(ch.FLAVOR_TAGS)),
}
_ENC_SPEC = None


def field_specs(channel):
    """Encounter heads come from the module vocabulary exported with the contexts."""
    global _ENC_SPEC
    if channel != "encounter_procurement@1":
        return FIELD_SPECS[channel]
    if _ENC_SPEC is None:
        vocab = ch.enc_vocab()
        _ENC_SPEC = {}
        for i in range(ch.ENC_MAX_ENEMIES):
            for cat in ch.ENC_CATEGORIES:
                _ENC_SPEC[f"e{i}_{cat}"] = ("choice", vocab[cat])
            _ENC_SPEC[f"e{i}_traits"] = ("multi", vocab["traits"])
    return _ENC_SPEC


def legal_for(channel, row, field, options):
    """Per-row legal mask, or None when every option is allowed."""
    if channel == "boss_tactic@1":
        return [row["options"][o]["legal"] for o in sim.OPTIONS]
    if channel == "music_scene@1":
        allowed = [str(v) for v in ch.music_legal(row["scene"])[field]]
        return [str(o) in allowed for o in options]
    if channel == "encounter_procurement@1" and not field.endswith("_traits"):
        allowed = ch.enc_allowed(row).get(field.split("_", 1)[1], {})
        return [o in allowed for o in options]
    if channel == "card_procurement@1":
        m = _CARD_FX.match(field)
        if m:
            j = int(m.group(1))
            pool, tier = row["spec"]["pool"], int(row["spec"]["tier"])
            effect_max = ch.CARD_EFFECT_LIMITS[pool][1]
            if j >= effect_max:
                return [str(o) == "none" for o in options]
            allowed = set(ch.card_allowed(pool, tier)) | {"none"}
            return [str(o) in allowed for o in options]
    if channel == "music_score@1" and field == "prog0":
        return [str(o) == "0" for o in options]
    return None


def num(v):
    """Canonical number text shared with GDScript (local_decision_prompts.gd): ints as-is,
    floats rounded to 2 decimals with trailing zeros stripped."""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, str):
        return v
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer()):
        return str(int(v))
    # Integer rounding (half away from zero) so GDScript's formatter cannot disagree on x.xx5 ties.
    n = math.floor(abs(float(v)) * 100 + 0.5 + 1e-7)
    s = f"{n // 100}.{n % 100:02d}".rstrip("0").rstrip(".")
    return ("-" + s) if v < 0 and n != 0 else s


def kv(d):
    """Sorted key=value pairs with zero values dropped (deterministic in both runtimes)."""
    return " ".join(f"{k}={num(d[k])}" for k in sorted(d) if d[k]) or "none"


def render_adaptation(row):
    c = row["comparison"]
    a = c["current_battle_assessment"]
    e = a["stage_expected"]
    hist = " ".join(f"b{num(h['requested_after_battle'])}:{num(h['challenge_delta'])}/{h['trait_id']}"
                    for h in row["history"][-3:]) or "none"
    return "\n".join([
        "<decision> difficulty_adaptation@1",
        f"battle={num(row['battle'])} stage_kind={e['kind']} expected_turns={num(e['turns'])} expected_damage={num(e['damage'])} applied_challenge={num(e['challenge_delta'])}",
        f"grade={a['grade']} skill={num(a['normalized_skill'])} turn_dev={num(a['turn_deviation'])} damage_dev={num(a['damage_deviation'])}",
        "components " + kv(a["components"]),
        f"elo={num(a['elo']['before'])}->{num(a['elo']['after'])} rating={num(c['skill_rating'])} skill_grade={c['skill_grade']}",
        f"optimal_prox={num(c['optimal_proximity'])} elo_prox={num(c['elo_proximity'])} seed_curve_prox={num(c['seed_curve_proximity'])} confidence={num(c['confidence'])} route={c['route']}",
        f"dialogue assisted={num(a['dialogue_context']['assisted'])} burdened={num(a['dialogue_context']['burdened'])}",
        f"history={hist}",
        "<decide>",
    ])


def render_nemesis(row):
    s = row["snapshot"]
    ep, ws = s["elite_priority"], s["weighted_signals"]
    return "\n".join([
        "<decision> nemesis_director@1",
        f"battles={num(s['battles_cleared'])} hp={num(s['hp'])}/{num(s['max_hp'])} int={num(s['int'])} deck_size={num(s['deck_size'])}",
        "deck " + kv(s["deck_tags"]),
        "choices " + kv(s["choices"]),
        f"totals turns={num(s['turns'])} cast={num(s['cards_cast'])} damage_taken={num(s['damage_taken'])}",
        f"elite turns={num(ep['turns'])} cast={num(ep['cards_cast'])} damage_taken={num(ep['damage_taken'])} terrain={num(ep['terrain_cards_cast'])} lowest_hp={num(ep['lowest_hp_after'])}",
        f"weighted turns={num(ws['turns'])} cast={num(ws['cards_cast'])} damage_taken={num(ws['damage_taken'])} terrain={num(ws['terrain_cards_cast'])}",
        "reactions " + kv(ws["reaction_counts"]),
        "<decide>",
    ])


def render_boss(row):
    """Canonical boss prompt (mirrored by local_decision_prompts.gd render_boss)."""
    s, opts = row["state"], row["options"]
    legal = [o for o in sim.OPTIONS if opts[o]["legal"]]
    word = lambda o: features.LABEL_WORDS[o][0].strip() if o in features.LABEL_WORDS else o
    opt_num = lambda v: "none" if v is None else num(v)
    lines = [
        "<decision> boss_tactic@1",
        "options " + " ".join(word(o) for o in legal),
        f"phase={num(s['phase'])} turn={num(s['turn'])} arena={s['arena']}",
        f"boss hp={num(s['boss_hp_pct'])} block={num(s['boss_block'])} at={num(s['boss_cell'][0])},{num(s['boss_cell'][1])}",
        f"player hp={num(s['player_hp_pct'])} block={num(s['player_block'])} chill={num(s['player_chill'])} at={num(s['player_cell'][0])},{num(s['player_cell'][1])}",
        f"dist={num(s['dist'])} debt={num(s['threat_debt'])} rhythm={word(s['fsm_suggestion'])}",
        "moves " + (" ".join(s["player_last_moves"]) or "none"),
        "dists " + (" ".join(num(d) for d in s["player_last_dists"]) or "none"),
        f"hits={num(s['player_hits_last3'])} stayed={opt_num(s['player_stayed_rate'])} blocked={opt_num(s['player_block_rate'])}",
        "boss_last " + (" ".join(word(x) for x in s["boss_last3"]) or "none"),
    ]
    for o in legal:
        f = opts[o]
        lines.append(f"{word(o)} dmg={num(f['dmg'])} area={num(f['area'])} safe={num(f['safe_cells'])} "
                     f"safe_in_range={num(f['safe_in_range'])} recent={num(f['used_last2'])}")
    lines.append("<decide>")
    return "\n".join(lines)


def render_fate(row):
    m, local = row["manifest"], row["local"]
    pool = {f"{tag}_{tier}": n for tag, tiers in m["reward_pool"].items() for tier, n in tiers.items()}
    lines = ["<decision> fate_analysis@1",
             "totals " + kv(m["totals"]),
             "pool " + kv(pool),
             "factors " + kv(m["route_seed_factors"]),
             "anchor " + " ".join(f"{r}={num(local['route_references'][r]['skilled'])}/{num(local['route_references'][r]['near_optimal'])}"
                                  for r in ch.FATE_ROUTES),
             f"bias={num(local['difficulty_bias'])}"]
    for e in m["encounters"]:
        lines.append(f"b{num(e['battle'])} {e['kind']} p={num(e['pressure'])} n={num(e['enemy_count'])} r={num(e['ranged_count'])} "
                     f"x={num(e['blocked_cells'])} t={num(e['trait_count'])}")
    lines.append("<decide>")
    return "\n".join(lines)


def render_music(row):
    return "\n".join(["<decision> music_scene@1", "scene " + kv(row["scene"]), "<decide>"])


def render_card(ctx):
    spec = ctx["spec"]
    pool, tier = spec["pool"], int(spec["tier"])
    limits = ctx.get("effect_limits", {})
    fx = ch.card_allowed(pool, tier)
    return "\n".join([
        "<decision> card_procurement@1",
        f"pool={pool} tier={num(tier)} cards={num(ch.card_count(ctx))} budget={num(ctx.get('budget', 0))} "
        f"effects={num(limits.get('min', 1))}..{num(limits.get('max', 1))}",
        "fx " + (" ".join(fx) if fx else "none"),
        "deck " + kv(ctx.get("deck_tags", {})),
        "<decide>",
    ])


def render_music_score(ctx):
    plan = ctx["plan"]
    scene = ctx["scene"]
    return "\n".join([
        "<decision> music_score@1",
        f"style={plan['style_id']} mode={plan['mode']} context={scene.get('context', 'origin')} "
        f"battle={num(scene.get('battle', 1))} density={plan['density_profile']} rhythm={plan['rhythm_profile']}",
        "<decide>",
    ])


def render_camp(ctx):
    b = ctx["battle"]
    comp = b.get("components", {}) if isinstance(b.get("components"), dict) else {}
    return "\n".join([
        "<decision> camp_assessment@1",
        f"score={num(b.get('battle_score', 50))} tempo={num(comp.get('tempo', 0.5))} "
        f"survival={num(comp.get('survival', 0.5))} tactics={num(comp.get('tactics', 0.5))}",
        "<decide>",
    ])


def render_flavor(ctx):
    return "\n".join([
        "<decision> encounter_flavor@1",
        f"stage={num(ctx.get('stage', 1))} kind={ctx.get('kind', 'normal')}",
        "<decide>",
    ])


def render_encounter(row):
    c = row["context"]
    k = c["contract"]
    snap = c.get("player_snapshot", {})
    formation = (k.get("formation_archetype") or {}).get("id", "none")
    lines = ["<decision> encounter_procurement@1",
             f"stage={num(row['stage'])} kind={k['kind']} enemies={num(k['enemy_count_required'])} budget={num(c['budget']['total_budget'])} "
             f"floor={num(k.get('spend_floor', 0.6))} trait_max={num(k.get('trait_max', 0))} counter_cap={num(k.get('counter_cap', 0))} "
             f"lock_cap={num(k.get('lock_cap', 0))} master={num(c.get('master_level', 0))}",
             "families " + (" ".join(sorted(k.get("required_attack_families", []))) or "none"),
             f"formation={formation}",
             "identity " + (" ".join(k.get("identity_skill_pool", [])) or "none"),
             f"player hp={num(snap.get('hp_ratio', 1))} deck={num(snap.get('deck_size', 0))}",
             "deck " + kv(snap.get("deck_tags", {})),
             "<decide>"]
    return "\n".join(lines)


RENDER = {"boss_tactic@1": render_boss, "difficulty_adaptation@1": render_adaptation,
          "nemesis_director@1": render_nemesis, "fate_analysis@1": render_fate,
          "music_scene@1": render_music, "encounter_procurement@1": render_encounter,
          "card_procurement@1": render_card, "music_score@1": render_music_score,
          "camp_assessment@1": render_camp, "encounter_flavor@1": render_flavor}
TEACHER_FILES = {  # teacher name -> (file pattern, label key); {stem} is the channel name without version
    "s1": ("out/s1_{stem}.jsonl", "s1"),
    "rule": ("out/rule_{stem}.jsonl", "rule"),
}


def _norm(d):
    return {str(k): float(v) for k, v in (d or {}).items()}


def targets_from_label(channel, row, lab):
    """One teacher's label -> {field: dist list} for the fields present (None if unusable)."""
    if not lab:
        return None
    if channel == "boss_tactic@1":
        dist = _norm(lab.get("dist"))
        return {"tactic": [dist.get(o, 0.0) for o in sim.OPTIONS]} if dist else None
    if not lab.get("n_valid"):
        return None
    out = {}
    for field, (kind, options) in field_specs(channel).items():
        if field not in lab:
            continue   # e.g. enemy slots beyond this stage's enemy count
        d = _norm(lab.get(field))
        out[field] = [d.get(str(o), 0.0) for o in options]
    return out or None


def ensemble(channel, row, per_teacher):
    """Average teachers field by field; choice fields are renormalised (inside the legal mask) first."""
    out = {}
    for field, (kind, options) in field_specs(channel).items():
        legal = legal_for(channel, row, field, options)
        acc, n = [0.0] * len(options), 0
        for t in per_teacher:
            if field not in t:
                continue
            v = t[field]
            if kind == "choice":
                if legal:
                    v = [x if ok else 0.0 for x, ok in zip(v, legal)]
                s = sum(v)
                if s <= 0:
                    continue
                v = [x / s for x in v]
            acc = [a + x for a, x in zip(acc, v)]
            n += 1
        if n:
            out[field] = {"kind": kind, "options": [str(o) for o in options], "dist": [a / n for a in acc]}
            if legal and kind == "choice":
                out[field]["legal"] = legal
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/sft_v1.jsonl")
    ap.add_argument("--teachers", default="s1,rule,qwen3_8-flash",
                    help="comma list: s1 = decision model, rule = local rule labels, else chat-teacher file tag")
    ap.add_argument("--boss-pikl", action="store_true", help="also average the simulator's piKL teacher into boss labels")
    ap.add_argument("--max-train-per-channel", type=int, default=8000,
                    help="cap train rows per channel so one channel cannot dominate training time")
    args = ap.parse_args()
    teachers = args.teachers.split(",")
    counts = {}
    with open(args.out, "w", encoding="utf-8") as f:
        for channel in RENDER:
            stem = channel.split("@")[0]
            merged = {}   # key -> (row, [targets...], [teacher names])
            for teacher in teachers:
                pattern, label_key = TEACHER_FILES.get(teacher, (f"out/llm_{{stem}}_{teacher}.jsonl", "llm"))
                for path in glob.glob(pattern.format(stem=stem)):
                    for line in open(path, encoding="utf-8"):
                        row = json.loads(line)
                        k = ch.CHANNELS[channel].key(row)
                        tt = targets_from_label(channel, row, (row.get("labels") or {}).get(label_key))
                        if tt is None:
                            continue
                        entry = merged.setdefault(k, (row, [], []))
                        entry[1].append(tt)
                        entry[2].append(teacher)
                        if args.boss_pikl and channel == "boss_tactic@1" and "pikl" not in entry[2]:
                            entry[1].append(targets_from_label(channel, row, row["labels"]["pikl"]))
                            entry[2].append("pikl")
            train_seen = 0
            for k, (row, tts, names) in merged.items():
                if row["split"] == "train":
                    train_seen += 1
                    if train_seen > args.max_train_per_channel:
                        continue
                targets = ensemble(channel, row, tts)
                if not targets:
                    continue
                f.write(json.dumps({"channel": channel, "split": row["split"], "key": k, "prompt": RENDER[channel](row),
                                    "targets": targets, "teachers": names}, ensure_ascii=False) + "\n")
                counts[(channel, row["split"])] = counts.get((channel, row["split"]), 0) + 1
    for k in sorted(counts):
        print(k, counts[k])
    print("->", args.out)


if __name__ == "__main__":
    main()
