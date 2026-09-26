"""Typed decision channels that replace the game's DeepSeek calls (structured in -> structured out).

Each channel provides:
  generate(n, rng)          synthetic input states (game-shaped payloads, split per sample)
  messages(row)             chat messages for an LLM teacher (system prompt copied from the game)
  parse(content, row)       validated structured label or None (same whitelist/clamps as the game)
  aggregate(samples)        per-field soft labels from several teacher samples

Free text fields (player_message, title, summary) are deliberately not labelled: the shipped
model picks typed fields and the game fills text from templates keyed by those fields.
"""
from __future__ import annotations

import json
import random
import re
import zlib
from collections import Counter

import features
import sim


def _split(uid):
    b = zlib.crc32(str(uid).encode()) % 10
    return "test" if b == 9 else ("val" if b == 8 else "train")


def _json_obj(content):
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.S)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _field_dists(samples, fields, multi=()):
    out = {}
    for f in fields:
        c = Counter(json.dumps(s[f]) for s in samples)
        out[f] = {json.loads(k): round(v / len(samples), 4) for k, v in c.items()} if f not in multi else None
    for f in multi:
        tags = Counter(t for s in samples for t in s[f])
        out[f] = {t: round(v / len(samples), 4) for t, v in tags.items()}
    return out


# ------------------------------------------------------------------ boss_tactic@1

class BossTactic:
    name = "boss_tactic@1"
    SYSTEM = (
        "You choose the next telegraphed action of the final boss in a turn-based 8x6 grid card battler.\n"
        "Rules: every boss skill locks onto the player's current cell and is shown one turn ahead; "
        "the player then moves up to 3 cells and casts. Terrain from skills burns for 2 turns.\n"
        "Design criteria: counter the player's repeated movement habits; keep tactical variety; "
        "prefer actions that force the player to trade attack range for safety; ease off when the "
        "player is nearly dead; follow the boss rhythm unless there is a good reason not to; "
        "use only public information shown below.\n"
        "Answer with ONE json object and nothing else: {\"choice\": \"<one option word>\"}. "
        "The option word must be one of the words listed after <options>."
    )

    def key(self, row):
        return f"{row['game']}:{row['turn']}"

    def messages(self, row):
        return [{"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": features.render_prompt(row["state"], row["options"])}]

    def parse(self, content, row):
        allowed = {features.LABEL_WORDS[o][0].strip(): o for o in sim.OPTIONS if row["options"][o]["legal"]}
        obj = _json_obj(content)
        if not obj:
            return None
        choice = allowed.get(str(obj.get("choice", "")).strip().lower())
        return {"choice": choice} if choice else None

    def aggregate(self, samples):
        c = Counter(s["choice"] for s in samples)
        return {"dist": {a: round(v / len(samples), 4) for a, v in c.items()}}


# ------------------------------------------------------------------ difficulty_adaptation@1
# Mirrors scripts/director/fate_model.gd build_adaptation_request / validate_adaptation.

ADAPT_TRAITS = ["none", "pursuit", "adaptive_guard", "status_reactor"]
ADAPT_REASONS = ["above_seed_curve", "below_seed_curve", "high_survival", "low_survival",
                 "fast_clears", "high_attrition", "strong_reactions", "stable_build", "uncertain_signal"]
ROUTES = {"fire": "火焰", "lightning": "雷电", "fire_water": "火水反应", "water_lightning": "水雷反应",
          "terrain_control": "地形控制", "utility_hybrid": "技巧混合"}
GRADES = ["S", "A", "B", "C", "D"]
ADAPT_SYSTEM = (  # verbatim from fate_model.gd:613
    "你是卡牌战棋游戏的异步难度导演。每场战斗后提出一次、且只作用于下一场尚未开始战斗的调整。"
    "comparison.current_battle_assessment 是本地真值层按本关难度归一化后的成绩：stage_expected.turns/damage 是该关预期，"
    "turn_deviation/damage_deviation 是实际减预期，normalized_skill 为0到1，grade为S到D，elo.before/after为逐战等级分；"
    "dialogue_context明确记录本战携带的对话增益和负面代价，只用于解释成绩与调整置信度，不得自行重算或篡改ELO。"
    "先看这些逐战字段，再结合命运曲线、精英双倍权重和历史调整；不要把不同关卡的原始回合或承伤直接比较。"
    "只能在安全边界内返回数值，不能创造新机制，不能元素免疫，不能修改玩家牌组。只输出JSON对象，不要Markdown。"
    "字段严格为：challenge_delta(-2到5整数)、hp_pct(-0.10到0.30)、attack_delta(-1到2整数)、block_delta(-2到6整数)、"
    "move_delta(-1到2整数)、trait_id(none/pursuit/adaptive_guard/status_reactor)、reason_tags(白名单数组)、"
    "player_message(最多60汉字)。优秀玩家优先增加阵型/机制压力，不要只堆生命；信号不确定时保持温和。"
)
CAPS = {  # validate_adaptation: per challenge_delta ceilings (hp, atk, block, move, trait allowed)
    1: (0.08, 0, 2, 0, False), 2: (0.14, 1, 3, 0, True), 3: (0.20, 1, 4, 1, True), 4: (0.25, 2, 5, 1, True),
}


def validate_adaptation(v):
    """Python port of fate_model.gd validate_adaptation; returns the clamped plan or None."""
    fields = ["challenge_delta", "hp_pct", "attack_delta", "block_delta", "move_delta"]
    if any(_num(v.get(f)) is None for f in fields):
        return None
    if v.get("trait_id") not in ADAPT_TRAITS or not isinstance(v.get("reason_tags", []), list):
        return None
    reasons = []
    for r in v.get("reason_tags", []):
        if r not in ADAPT_REASONS:
            return None
        if r not in reasons:
            reasons.append(r)
    clamp = lambda x, lo, hi: max(lo, min(hi, x))
    plan = {
        "challenge_delta": clamp(int(round(_num(v["challenge_delta"]))), -2, 5),
        "hp_pct": round(clamp(_num(v["hp_pct"]), -0.10, 0.30), 2),
        "attack_delta": clamp(int(round(_num(v["attack_delta"]))), -1, 2),
        "block_delta": clamp(int(round(_num(v["block_delta"]))), -2, 6),
        "move_delta": clamp(int(round(_num(v["move_delta"]))), -1, 2),
        "trait_id": v["trait_id"],
        "reason_tags": reasons[:3],
    }
    cd = plan["challenge_delta"]
    if cd <= 0:
        plan.update(hp_pct=min(0.0, plan["hp_pct"]), attack_delta=min(0, plan["attack_delta"]),
                    block_delta=min(0, plan["block_delta"]), move_delta=min(0, plan["move_delta"]), trait_id="none")
    elif cd in CAPS:
        hp, atk, blk, mv, trait_ok = CAPS[cd]
        plan.update(hp_pct=min(hp, plan["hp_pct"]), attack_delta=min(atk, plan["attack_delta"]) if atk else 0,
                    block_delta=min(blk, plan["block_delta"]), move_delta=min(mv, plan["move_delta"]) if mv else 0)
        if not trait_ok:
            plan["trait_id"] = "none"
    plan["hp_bucket"] = int(round(plan["hp_pct"] * 50))  # 0.02 steps: -5..15
    return plan


class DifficultyAdaptation:
    name = "difficulty_adaptation@1"
    FIELDS = ["challenge_delta", "hp_bucket", "attack_delta", "block_delta", "move_delta", "trait_id"]

    def key(self, row):
        return row["uid"]

    def generate(self, n, rng: random.Random):
        """Game-shaped synthetic comparisons driven by a latent skill, with run history."""
        rows = []
        for i in range(n):
            skill = rng.betavariate(2.2, 2.0)                      # latent player strength 0..1
            battle = rng.randint(1, 10)
            noise = lambda s: rng.gauss(0, s)
            norm = min(1.0, max(0.0, skill + noise(0.12)))
            grade = GRADES[min(4, max(0, int((1 - norm) * 5)))]
            expected_turns = round(rng.uniform(4.5, 8.0), 2)
            expected_dmg = round(rng.uniform(5.0, 14.0), 2)
            turn_dev = round((0.5 - norm) * 4 + noise(0.8), 2)
            dmg_dev = round((0.5 - norm) * 12 + noise(2.5), 2)
            elo_before = int(min(1700, max(650, 900 + 500 * skill + noise(80))))
            delta = int((norm - 0.5) * 60 + noise(10))
            route = rng.choice(list(ROUTES))
            prox = round(max(0.2, min(1.25, 0.35 + 0.65 * skill + noise(0.06))), 2)
            assisted = rng.random() < 0.3
            burdened = rng.random() < 0.15
            history = []
            level = 0
            for b in range(max(0, battle - 3), battle):
                level = max(-2, min(5, level + rng.choice([-1, 0, 0, 1, 1]) if b else 0))
                history.append({"requested_after_battle": b + 1, "challenge_delta": level,
                                "trait_id": rng.choice(ADAPT_TRAITS) if level >= 2 else "none"})
            assessment = {
                "components": {"survival": round(min(1, max(0, norm + noise(0.15))), 2),
                               "tactics": round(min(1, max(0, norm + noise(0.2))), 2),
                               "tempo": round(min(1, max(0, norm + noise(0.15))), 2)},
                "damage_deviation": dmg_dev, "turn_deviation": turn_dev,
                "dialogue_context": {"assisted": assisted, "burdened": burdened},
                "elo": {"before": elo_before, "after": elo_before + delta, "delta": delta},
                "grade": grade, "normalized_skill": round(norm, 2),
                "stage_expected": {"stage": battle, "turns": expected_turns, "damage": expected_dmg,
                                   "kind": "elite" if battle in (3, 7) else ("boss" if battle == 10 else "normal"),
                                   "challenge_delta": history[-1]["challenge_delta"] if history else 0},
            }
            comparison = {
                "confidence": round(min(0.95, 0.4 + 0.06 * battle + noise(0.05)), 2),
                "current_battle_assessment": assessment,
                "optimal_proximity": prox, "elo_proximity": round(max(0, min(1.2, skill + noise(0.1))), 2),
                "seed_curve_proximity": round(max(0.3, min(1.5, 0.6 + 0.7 * skill + noise(0.1))), 2),
                "route": route, "route_name": ROUTES[route], "skill_grade": grade,
                "skill_rating": elo_before + delta,
            }
            uid = f"adapt:{rng.getrandbits(40):x}"
            rows.append({"rec": "decision", "schema": self.name, "uid": uid, "split": _split(uid),
                         "battle": battle, "comparison": comparison, "history": history,
                         "latent_skill": round(skill, 3)})  # latent_skill: analysis only, never a model input
        return rows

    def payload(self, row):
        return {"comparison": row["comparison"],
                "current_battle_assessment": row["comparison"]["current_battle_assessment"],
                "previous_adjustments": row["history"],
                "bounds": {"challenge_delta": [-2, 5], "hp_pct": [-0.10, 0.30], "attack_delta": [-1, 2],
                           "block_delta": [-2, 6], "move_delta": [-1, 2], "trait_ids": ADAPT_TRAITS,
                           "reason_tags": ADAPT_REASONS}}

    def messages(self, row):
        return [{"role": "system", "content": ADAPT_SYSTEM},
                {"role": "user", "content": json.dumps(self.payload(row), ensure_ascii=False)}]

    def parse(self, content, row):
        obj = _json_obj(content)
        return validate_adaptation(obj) if obj else None

    def aggregate(self, samples):
        return _field_dists(samples, self.FIELDS, multi=("reason_tags",)) | {
            "plans": [{k: s[k] for k in self.FIELDS + ["hp_pct", "reason_tags"]} for s in samples]}


# ------------------------------------------------------------------ nemesis_director@1
# Mirrors scripts/director/nemesis_planner.gd make_snapshot / build_chat_request / validate_recipe.

NEMESIS_PLAYER_TAGS = ["fire_specialist", "lightning_specialist", "utility_planner", "terrain_tactician", "risk_taker",
                       "cautious_survivor", "card_collector", "deck_curator", "scholar", "explorer", "burst_caster",
                       "attrition_caster", "low_defense"]
NEMESIS_ARCHETYPES = ["furnace_examiner", "storm_mirror", "arcane_archivist", "crown_hunter"]
NEMESIS_MECHANICS = ["pursuit", "fortified", "heavy_strikes", "adaptive_guard", "status_reactor", "storm_cycle"]
NEMESIS_PHASES = ["accelerating_exam", "guard_break_exam", "steady_exam"]
NEMESIS_ARENAS = ["open_crown", "split_forge", "narrow_ring"]
NEMESIS_ULTIMATES = ["phoenix_thesis", "storm_coronation", "perfect_preparation", "apprentice_finale"]
NEMESIS_SYSTEM = (  # verbatim from nemesis_planner.gd:148 (title/summary kept so the teacher answers normally)
    "你是卡牌战棋游戏的宿敌导演。你只能从白名单 ID 中选择内容，禁止创造新 ID、脚本、数值或免疫机制。"
    "分析时必须优先参考 elite_priority；它来自两场精英战，并且 weighted_signals 已将精英表现按2倍权重计算。"
    "Boss 应回应玩家习惯但不能废掉玩家构筑；必杀技应强化玩家已形成的方向。只输出一个 JSON 对象，不要 Markdown。"
    "输出字段必须严格为：player_tags(2到4个ID)、boss{archetype, mechanic_tags(恰好2个ID), phase_tag, arena_tag}、"
    "ultimate_card、title(最多12个汉字)、summary(最多60个汉字)。"
)
BUILDS = {  # latent build -> deck tag weights and typical reactions
    "fire": ({"Fire": 6, "Water": 1, "Utility": 2}, ["burn_spread"]),
    "lightning": ({"Lightning": 6, "Water": 1, "Utility": 2}, ["chain_shock"]),
    "fire_water": ({"Fire": 4, "Water": 4, "Utility": 1}, ["fire_water"]),
    "water_lightning": ({"Water": 4, "Lightning": 4, "Utility": 1}, ["water_lightning"]),
    "terrain": ({"Fire": 2, "Water": 2, "Frost": 2, "Nature": 2, "Utility": 2}, ["fire_water", "frost_shatter"]),
    "utility": ({"Utility": 5, "Arcane": 3, "Fire": 1}, []),
}


def validate_recipe(v):
    """Python port of nemesis_planner.gd validate_recipe (typed fields only)."""
    tags = v.get("player_tags", [])
    boss = v.get("boss", {})
    if not isinstance(tags, list) or not isinstance(boss, dict) or not 2 <= len(tags) <= 4:
        return None
    mech = boss.get("mechanic_tags", [])
    if any(t not in NEMESIS_PLAYER_TAGS for t in tags) or not isinstance(mech, list) or len(mech) != 2 \
            or any(m not in NEMESIS_MECHANICS for m in mech):
        return None
    if boss.get("archetype") not in NEMESIS_ARCHETYPES or boss.get("phase_tag") not in NEMESIS_PHASES \
            or boss.get("arena_tag") not in NEMESIS_ARENAS or v.get("ultimate_card") not in NEMESIS_ULTIMATES:
        return None
    return {"player_tags": list(dict.fromkeys(tags)), "archetype": boss["archetype"], "mechanic_tags": list(mech),
            "phase_tag": boss["phase_tag"], "arena_tag": boss["arena_tag"], "ultimate_card": v["ultimate_card"]}


class NemesisDirector:
    name = "nemesis_director@1"
    FIELDS = ["archetype", "phase_tag", "arena_tag", "ultimate_card"]

    def key(self, row):
        return row["uid"]

    def generate(self, n, rng: random.Random):
        rows = []
        for _ in range(n):
            build = rng.choice(list(BUILDS))
            weights, reactions = BUILDS[build]
            risk = rng.random()        # 0 cautious .. 1 risky
            curation = rng.random()    # 0 collector .. 1 curator
            study = rng.random()
            battles = rng.randint(8, 10)
            deck_size = int(max(10, min(26, 14 + (1 - curation) * 8 + rng.gauss(0, 2))))
            deck_tags = {t: 0 for t in ("Basic", "Fire", "Water", "Frost", "Lightning", "Arcane", "Nature", "Utility")}
            deck_tags["Basic"] = int(max(1, round(6 - curation * 4 + rng.gauss(0, 1))))
            pool = [t for t, w in weights.items() for _ in range(w)]
            for _ in range(max(0, deck_size - deck_tags["Basic"])):
                deck_tags[rng.choice(pool)] += 1
            max_hp = rng.randint(30, 46)
            dmg_taken = int(max(4, 10 + risk * 30 + rng.gauss(0, 5)))
            turns = int(max(20, battles * (4.5 + (1 - risk) * 2) + rng.gauss(0, 4)))
            cards_cast = int(turns * (1.5 + risk * 0.8) + rng.gauss(0, 5))
            terrain_cast = int(max(0, (8 if build == "terrain" else 2) + rng.gauss(0, 2)))
            reaction_counts = {r: max(0, int(rng.gauss(4, 2))) for r in reactions}
            elite_turns = int(max(6, 11 + (1 - risk) * 4 + rng.gauss(0, 2)))
            elite_dmg = int(max(0, dmg_taken * 0.4 + rng.gauss(0, 2)))
            elite_cast = int(elite_turns * (1.6 + risk * 0.6))
            elite_terrain = int(max(0, terrain_cast * 0.4))
            elite_react = {r: max(0, v // 2) for r, v in reaction_counts.items()}
            choices = {"train": int(study * 4 + rng.random() * 1.5), "rest": int((1 - risk) * 3 + rng.random()),
                       "explore": int(rng.random() * 3), "skip_reward": int(curation * 3 + rng.random()),
                       "remove_basic": int(curation * 2 + rng.random()), "card_reward": int((1 - curation) * 6 + 2)}
            hp = int(max(1, min(max_hp, max_hp * (0.95 - risk * 0.5) + rng.gauss(0, 3))))
            snapshot = {
                "schema_version": 2, "battles_cleared": battles, "turns": turns, "cards_cast": cards_cast,
                "damage_taken": dmg_taken, "hp": hp, "max_hp": max_hp, "hp_ratio": round(hp / max_hp, 2),
                "int": rng.randint(1, 5), "deck_size": sum(deck_tags.values()), "deck_tags": deck_tags,
                "choices": choices,
                "elite_priority": {"battles": 2, "turns": elite_turns, "cards_cast": elite_cast, "damage_taken": elite_dmg,
                                   "terrain_cards_cast": elite_terrain, "reaction_counts": elite_react,
                                   "lowest_hp_after": int(max(1, max_hp * (0.8 - risk * 0.5)))},
                "weighted_signals": {"turns": turns + elite_turns, "cards_cast": cards_cast + elite_cast,
                                     "damage_taken": dmg_taken + elite_dmg, "terrain_cards_cast": terrain_cast + elite_terrain,
                                     "reaction_counts": {r: reaction_counts[r] + elite_react[r] for r in reaction_counts}},
                "reaction_counts": reaction_counts,
            }
            uid = f"nemesis:{rng.getrandbits(40):x}"
            rows.append({"rec": "decision", "schema": self.name, "uid": uid, "split": _split(uid),
                         "run_seed": rng.getrandbits(31), "snapshot": snapshot,
                         "latent": {"build": build, "risk": round(risk, 2), "curation": round(curation, 2),
                                    "study": round(study, 2)}})  # latent: analysis only
        return rows

    def messages(self, row):
        contract = {"player_tags": NEMESIS_PLAYER_TAGS, "boss_archetypes": NEMESIS_ARCHETYPES,
                    "mechanic_tags": NEMESIS_MECHANICS, "phase_tags": NEMESIS_PHASES, "arena_tags": NEMESIS_ARENAS,
                    "ultimate_cards": NEMESIS_ULTIMATES}
        payload = {"run_seed": row["run_seed"], "player_snapshot": row["snapshot"], "allowlist": contract}
        return [{"role": "system", "content": NEMESIS_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]

    def parse(self, content, row):
        obj = _json_obj(content)
        return validate_recipe(obj) if obj else None

    def aggregate(self, samples):
        return _field_dists(samples, self.FIELDS, multi=("player_tags", "mechanic_tags")) | {"recipes": samples}


# ------------------------------------------------------------------ SystemOne decision-model questions
# decision-model-preview (/compatible-mode/v1/systemone) answers typed questions directly:
#   choice -> {choice, probabilities, confidence}; score -> probabilities over list indices; noul -> p in [0,1].
# The resulting distributions are stored in the same shape as the chat-teacher aggregates.

BOSS_OPTION_TEXT = {
    "siphon": "soul chain on the player's single cell; chills the player and gives the boss +2 block",
    "abyss_cross": "full row and full column through the player's cell",
    "nova": "3x3 square centred on the player's cell (red cells shown)",
    "eclipse_diagonal": "both full diagonals through the player's cell",
    "world_rend": "the entire row of the player's cell",
    "advance": "walk a locked path toward the player; hits only if the player stays on it",
    "move": "reposition without attacking; the next turn must attack",
}
NEMESIS_TEXT = {  # descriptions from nemesis_planner.gd
    "furnace_examiner": "熔炉考官：用清晰的大招周期检验爆发时机", "storm_mirror": "风暴镜像：快速换位，迫使玩家重新组织弹射路径",
    "arcane_archivist": "奥术典藏官：防守与进攻交替，检验牌组完整度", "crown_hunter": "王冠追猎者：持续逼近，检验移动和防御资源",
    "accelerating_exam": "加速终试：首领的大招周期缩短", "guard_break_exam": "攻守换卷：首领会周期性进入可被击破的防御回合",
    "steady_exam": "稳定终试：维持标准三回合首领节奏",
    "open_crown": "开放王冠：空间宽阔，适合主动调整站位", "split_forge": "分割熔炉：中央障碍改变追击与弹射路线",
    "narrow_ring": "收束法环：边缘障碍增加近身压力，但保留两条撤退路线",
    "phoenix_thesis": "凤凰终论：引爆全部燃烧并重新施加燃烧", "storm_coronation": "雷冠加冕：结算全部感电并再次震击所有敌人",
    "perfect_preparation": "完美备课：获得大量格挡、抽牌并恢复法力", "apprentice_finale": "学徒终章：对所有敌人造成稳定伤害并抽牌",
    "fire_specialist": "燃烧构筑：持续施加、扩散并结算燃烧", "lightning_specialist": "感电构筑：依靠感电、弹射和群体结算",
    "utility_planner": "技巧规划者：重视移动、防御、抽牌与费用", "terrain_tactician": "地形反应师：主动铺设地形并制造元素反应",
    "risk_taker": "高风险成长：带伤推进并优先换取成长", "cautious_survivor": "谨慎生存：会为持续生命主动安排休整",
    "card_collector": "法术收藏家：倾向把更多法术加入牌组", "deck_curator": "牌组修整者：愿意跳过奖励或移除基础牌",
    "scholar": "研习派：多次选择研习来强化已有法术", "explorer": "探索派：愿意放弃确定成长换取额外选牌",
    "burst_caster": "爆发施法者：用较少回合打出较多卡牌", "attrition_caster": "持久施法者：通过较长战斗逐步建立优势",
    "low_defense": "低防御倾向：承受伤害较多，偏向主动输出",
    "pursuit": "追猎步法：移动力+1", "fortified": "预制护甲：开场获得8点格挡", "heavy_strikes": "沉重施压：攻击力+1",
    "adaptive_guard": "适应屏障：防御回合获得更多格挡", "status_reactor": "元素反应装甲：带燃烧或感电时获得少量格挡",
    "storm_cycle": "充能轮转：周期性使用可预警的充能法术",
}
ADAPT_REASON_TEXT = {
    "above_seed_curve": "玩家表现高于本种子的参考曲线", "below_seed_curve": "玩家表现低于本种子的参考曲线",
    "high_survival": "玩家生存状况很好", "low_survival": "玩家生存吃紧", "fast_clears": "玩家清场明显快于预期",
    "high_attrition": "玩家承伤或消耗明显高于预期", "strong_reactions": "玩家大量打出元素反应",
    "stable_build": "玩家构筑已经稳定成型", "uncertain_signal": "信号不足或互相矛盾，应保持温和",
}
ADAPT_TRAIT_TEXT = {"none": "不附加特性", "pursuit": "追猎：敌人移动力+1", "adaptive_guard": "适应屏障：防御回合更多格挡",
                    "status_reactor": "元素反应装甲：带燃烧或感电时获得少量格挡"}
ADAPT_LEVELS = {  # score lists -> index maps back to the value
    "challenge_delta": [(-2, "明显降低难度"), (-1, "略微降低难度"), (0, "保持当前难度"), (1, "温和提高"),
                        (2, "提高实际采购强度"), (3, "启用完整攻守组合"), (4, "明显提高机制与阵型压力"), (5, "宗师级压力")],
    "hp_pct": [(-0.10, "敌人生命-10%"), (-0.05, "敌人生命-5%"), (0.0, "生命不变"), (0.05, "生命+5%"), (0.10, "生命+10%"),
               (0.15, "生命+15%"), (0.20, "生命+20%"), (0.25, "生命+25%"), (0.30, "生命+30%")],
    "attack_delta": [(-1, "攻击-1"), (0, "攻击不变"), (1, "攻击+1"), (2, "攻击+2")],
    "block_delta": [(v, f"格挡{v:+d}") for v in range(-2, 7)],
    "move_delta": [(-1, "移动-1"), (0, "移动不变"), (1, "移动+1"), (2, "移动+2")],
}


def _score_dist(answer, levels):
    probs = answer.get("probabilities", {})
    return {levels[int(i)][0]: float(p) for i, p in probs.items() if int(i) < len(levels)}


def boss_s1(row):
    legal = [o for o in sim.OPTIONS if row["options"][o]["legal"]]
    criteria = {o: f"{BOSS_OPTION_TEXT[o]}; damage {row['options'][o]['dmg']}, covers {row['options'][o]['area']} cells, "
                   f"leaves the player {row['options'][o]['safe_in_range']} safe cells still in attack range, "
                   f"used {row['options'][o]['used_last2']}x in the last 2 turns" for o in legal}
    state = {k: v for k, v in row["state"].items()}
    questions = {"tactic": {"type": "choice", "criteria": criteria, "instructions": BossTactic.SYSTEM.split("Answer with")[0]
                            + "Which telegraphed action should the boss take this turn? `fsm_suggestion` is the designed rhythm."}}
    return state, questions


def boss_s1_label(ans, row):
    return {"dist": {o: round(float(p), 4) for o, p in ans["tactic"]["probabilities"].items()},
            "confidence": ans["tactic"].get("confidence")}


def adapt_s1(row):
    guide = "你是卡牌战棋游戏的异步难度导演，为下一场尚未开始的战斗提出调整。优秀玩家优先增加阵型/机制压力，不要只堆生命；信号不确定时保持温和。"
    q = {f: {"type": "score", "instructions": f"{guide} 这一项应取哪一档？（{f}）", "criteria": [t for _, t in levels]}
         for f, levels in ADAPT_LEVELS.items()}
    q["trait_id"] = {"type": "choice", "instructions": f"{guide} 下一场敌人附加哪个特性？", "criteria": ADAPT_TRAIT_TEXT}
    # Mutually exclusive reasons are asked as one choice so they cannot both be "yes".
    q["reason_curve"] = {"type": "choice", "instructions": "玩家本战相对种子参考曲线的位置？",
                         "criteria": {"above_seed_curve": ADAPT_REASON_TEXT["above_seed_curve"],
                                      "below_seed_curve": ADAPT_REASON_TEXT["below_seed_curve"],
                                      "on_curve": "与参考曲线基本持平"}}
    q["reason_survival"] = {"type": "choice", "instructions": "玩家的生存状况？",
                            "criteria": {"high_survival": ADAPT_REASON_TEXT["high_survival"],
                                         "low_survival": ADAPT_REASON_TEXT["low_survival"], "normal": "生存状况一般"}}
    for tag in ("fast_clears", "high_attrition", "strong_reactions", "stable_build", "uncertain_signal"):
        q[f"reason__{tag}"] = {"type": "noul", "instructions": f"这次调整的理由是否包括：{ADAPT_REASON_TEXT[tag]}？"}
    return DifficultyAdaptation().payload(row), q


def adapt_s1_label(ans, row):
    out = {f: _score_dist(ans[f], levels) for f, levels in ADAPT_LEVELS.items()}
    hp = out.pop("hp_pct")
    buckets = {}
    for v, p in hp.items():
        b = int(round(v * 50))
        buckets[b] = buckets.get(b, 0.0) + p
    out["hp_bucket"] = buckets
    out["trait_id"] = {k: float(v) for k, v in ans["trait_id"]["probabilities"].items()}
    reasons = {t: float(ans[f"reason__{t}"]["noul"]) for t in
               ("fast_clears", "high_attrition", "strong_reactions", "stable_build", "uncertain_signal")}
    for q in ("reason_curve", "reason_survival"):
        for t, p in ans[q]["probabilities"].items():
            if t in ADAPT_REASONS:
                reasons[t] = float(p)
    out["reason_tags"] = {t: reasons.get(t, 0.0) for t in ADAPT_REASONS}
    out["n_valid"] = 1
    return out


def nemesis_s1(row):
    guide = ("你是卡牌战棋游戏的宿敌导演。优先参考 elite_priority；weighted_signals 已将精英表现按2倍权重计算。"
             "Boss 应回应玩家习惯但不能废掉玩家构筑；必杀技应强化玩家已形成的方向。")
    pick = lambda ids: {i: NEMESIS_TEXT[i] for i in ids}
    q = {"archetype": {"type": "choice", "instructions": guide + " 选择 Boss 原型。", "criteria": pick(NEMESIS_ARCHETYPES)},
         "phase_tag": {"type": "choice", "instructions": guide + " 选择终试阶段。", "criteria": pick(NEMESIS_PHASES)},
         "arena_tag": {"type": "choice", "instructions": guide + " 选择场地。", "criteria": pick(NEMESIS_ARENAS)},
         "ultimate_card": {"type": "choice", "instructions": guide + " 选择奖励给玩家的必杀技。", "criteria": pick(NEMESIS_ULTIMATES)}}
    for t in NEMESIS_PLAYER_TAGS:
        q[f"tag__{t}"] = {"type": "noul", "instructions": f"这名玩家是否符合画像「{NEMESIS_TEXT[t]}」？"}
    for m in NEMESIS_MECHANICS:
        q[f"mech__{m}"] = {"type": "noul", "instructions": guide + f" Boss 是否应该带机制「{NEMESIS_TEXT[m]}」（共选2个）？"}
    return row["snapshot"], q   # 23 questions: label_systemone splits requests at the 16-question limit


def nemesis_s1_label(ans, row):
    out = {f: {k: float(v) for k, v in ans[f]["probabilities"].items()}
           for f in ("archetype", "phase_tag", "arena_tag", "ultimate_card")}
    out["player_tags"] = {t: float(ans[f"tag__{t}"]["noul"]) for t in NEMESIS_PLAYER_TAGS}
    out["mechanic_tags"] = {m: float(ans[f"mech__{m}"]["noul"]) for m in NEMESIS_MECHANICS}
    out["n_valid"] = 1
    return out


# ------------------------------------------------------------------ fate_analysis@1
# Mirrors scripts/director/fate_model.gd. The local anchor (generate_local_analysis) computes every
# number; the model only proposes bounded corrections on top of it (the rulebook's own procedure:
# "先使用给定锚点计算，再做有界的机制修正"). The recommended route is then derived as the best corrected
# route, so it can never contradict the numbers.

FATE_ROUTES = ["fire", "lightning", "fire_water", "water_lightning", "terrain_control", "utility_hybrid"]
FATE_ROUTE_TEXT = {  # fate_rulebook().route_rules
    "fire": "Fire单系：施加Burn→扩散Burn→集中结算；群体持续压制强，依赖状态铺设时间",
    "lightning": "Lightning单系：施加Shock→站位/目标调整→弹射或集中结算；敌人聚集时强，分散时效率下降",
    "fire_water": "Fire+Water蒸汽反应：跨Tag桥接换取爆发与区域控制；任一元素组件稀少都会增加成型方差",
    "water_lightning": "Water+Lightning导电反应：先铺湿润/水地形再连锁Shock；上限高但有双组件税",
    "terrain_control": "用水/火/雷地形与强制位移控制格子；障碍形成瓶颈时受益，但过度阻塞也压缩施法线",
    "utility_hybrid": "技巧牌提供移动、格挡、抽牌、费用与少量元素桥接；稳定低方差但终结上限通常较低",
}
FATE_ADJ = [(-30, "比本地锚点下调约30分"), (-15, "下调约15分"), (0, "维持本地锚点"), (15, "上调约15分"), (30, "上调约30分")]
FATE_BIAS_SHIFT = [(-0.4, "比本地压力估计明显更容易"), (-0.2, "略容易"), (0.0, "与本地压力估计一致"), (0.2, "略难"), (0.4, "比本地压力估计明显更难")]
FATE_UNCERTAINTY = [(0.12, "输入充分，路线价值稳定"), (0.2, "一般"), (0.3, "路线价值较依赖抽牌"), (0.42, "组件稀少，高度依赖运气")]
FATE_FIELDS = [f"adj_{r}" for r in FATE_ROUTES] + ["bias_shift", "uncertainty"]


def _nearest(levels, x):
    return min(levels, key=lambda a: abs(a[0] - x))[0]


def fate_state(row):
    m, local = row["manifest"], row["local"]
    return {"totals": m["totals"], "reward_pool": {k: v for k, v in m["reward_pool"].items() if any(v.values())},
            "route_seed_factors": m["route_seed_factors"],
            "encounters": [{k: e[k] for k in ("battle", "kind", "pressure", "enemy_count", "ranged_count", "blocked_cells", "trait_count")}
                           for e in m["encounters"]],
            "local_anchor": {r: local["route_references"][r] for r in FATE_ROUTES},
            "local_difficulty_bias": local["difficulty_bias"]}


class FateAnalysis:
    name = "fate_analysis@1"
    SYSTEM = (
        "你是《虚妄余罪》的种子分析器。本地锚点公式已经算出六条路线的 skilled / near_optimal 与难度偏置；"
        "你的任务只是按规则做有界的机制修正，不要重新估算总分。\n"
        "路线规则：" + "；".join(f"{r}={t}" for r, t in FATE_ROUTE_TEXT.items()) + "。\n"
        "修正依据：各战远程数/特性数/障碍格/精英压力与奖励池组件机会（T1=1、T2=1.45、T3=2），以及路线方差。"
        "每条路线的修正必须在 -30 到 +30 之间，没有明确理由就给 0；不同路线应按各自的组件机会区别对待。\n"
        "只输出一个 JSON 对象，不要 Markdown：{\"adjust\": {路线ID: 整数修正}, \"bias_shift\": -0.4到0.4, "
        "\"uncertainty\": 0.1到0.45, \"reason\": \"不超过40字\"}"
    )

    def key(self, row):
        return str(row["run_seed"])

    def generate(self, n, rng):
        rows = [json.loads(l) for l in open("out/fate_manifests_v1.jsonl", encoding="utf-8")][:n]
        for r in rows:
            r.update(rec="decision", schema=self.name, split=_split(r["run_seed"]))
        return rows

    def messages(self, row):
        return [{"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": json.dumps(fate_state(row), ensure_ascii=False)}]

    def parse(self, content, row):
        v = _json_obj(content)
        adj = (v or {}).get("adjust")
        if not isinstance(adj, dict):
            return None
        out = {}
        for r in FATE_ROUTES:
            x = _num(adj.get(r))
            if x is None or not -30 <= x <= 30:
                return None
            out[f"adj_{r}"] = _nearest(FATE_ADJ, x)
        shift, unc = _num(v.get("bias_shift")), _num(v.get("uncertainty"))
        if shift is None or unc is None or not -0.4 <= shift <= 0.4 or not 0.05 <= unc <= 0.5:
            return None
        out["bias_shift"] = _nearest(FATE_BIAS_SHIFT, shift)
        out["uncertainty"] = _nearest(FATE_UNCERTAINTY, unc)
        return out

    def aggregate(self, samples):
        return _field_dists(samples, FATE_FIELDS)


def fate_s1(row):
    local = row["local"]["route_references"]
    q = {}
    for r in FATE_ROUTES:
        q[f"adj__{r}"] = {"type": "score", "criteria": [t for _, t in FATE_ADJ],
                          "instructions": f"路线{r}（{FATE_ROUTE_TEXT[r]}）的本地锚点 near_optimal={local[r]['near_optimal']}。"
                                          "结合各战远程/特性/障碍/精英压力与奖励池中该路线组件的机会，做有界修正；没有明确理由就维持。"}
    q["bias_shift"] = {"type": "score", "criteria": [t for _, t in FATE_BIAS_SHIFT],
                       "instructions": f"本地压力公式给出的 difficulty_bias={row['local']['difficulty_bias']}（-1更容易，+1更难）。"
                                       "结合远程、特性与精英压力，这个估计需要怎样修正？"}
    q["uncertainty"] = {"type": "score", "criteria": [t for _, t in FATE_UNCERTAINTY],
                        "instructions": "这一局路线价值估计的不确定度？"}
    return fate_state(row), q


def fate_s1_label(ans, row):
    out = {f"adj_{r}": _score_dist(ans[f"adj__{r}"], FATE_ADJ) for r in FATE_ROUTES}
    out["bias_shift"] = _score_dist(ans["bias_shift"], FATE_BIAS_SHIFT)
    out["uncertainty"] = _score_dist(ans["uncertainty"], FATE_UNCERTAINTY)
    out["n_valid"] = 1
    return out


FateAnalysis.s1, FateAnalysis.s1_label = staticmethod(fate_s1), staticmethod(fate_s1_label)


# ------------------------------------------------------------------ music_scene@1
# Mirrors scripts/sound/scene_composer.gd. Mode stays owned by the deterministic director and the
# melody by the local procedural score. The model picks arrangement parameters, but only from the
# options the scene's design rules allow (legal sets), so it cannot contradict the director.

MUSIC_STYLES = {
    "apprentice_theme": "学徒主题：轻快、叙事性，适合开局与普通剧情",
    "ember_march": "余烬进行曲：火系路线，推进感强的小调进行曲",
    "storm_counterpoint": "风暴对位：雷系路线，快速对位与切分",
    "terrain_clockwork": "地形发条：地形/技巧路线，机械切分与低音锚点",
    "safe_archive": "安全档案馆：探索与安全场景，舒缓留白",
    "nemesis_exam": "宿敌终试：Boss 战，庄严压迫的终试主题",
    "crown_resolution": "王冠终章：胜利与正向结局，开阔的收束",
}
MUSIC_DENSITY = {"sparse": "稀疏，留白多，不压对白", "balanced": "均衡", "dense": "密集，全声部推进"}
MUSIC_RHYTHM = {"minimal": "极简鼓组", "steady": "稳定律动", "driving": "驱动型推进", "relentless": "不间断高压"}
MUSIC_BPM = [(-4, "放慢一点"), (0, "保持"), (4, "加快一点")]
MUSIC_TENSION = [(-0.10, "降低张力"), (0.0, "保持"), (0.10, "提高张力")]
MUSIC_CONTEXTS = ["origin", "battle", "elite", "boss", "story", "explore", "intermission", "reward", "victory", "defeat"]
MUSIC_BATTLE_STYLES = ["ember_march", "storm_counterpoint", "terrain_clockwork", "apprentice_theme"]
MUSIC_FIELDS = ["style_id", "density_profile", "rhythm_profile", "bpm_shift", "tension_shift"]


def music_story_style(scene):
    """scene_composer.suggest_style for story scenes (the director enforces it)."""
    if scene.get("story_position") == "ending":
        return "crown_resolution"
    if scene.get("story_position") == "post_battle" or float(scene.get("story_tension", 0.35)) <= 0.30:
        return "safe_archive"
    return "apprentice_theme"


def music_legal(scene):
    """Design rules per context -> allowed options for each field (mirrored in local_decision_policy.gd)."""
    c = scene["context"]
    style = {"boss": ["nemesis_exam"], "explore": ["safe_archive"], "victory": ["crown_resolution"],
             "story": [music_story_style(scene)], "battle": MUSIC_BATTLE_STYLES, "elite": MUSIC_BATTLE_STYLES,
             "defeat": MUSIC_BATTLE_STYLES}.get(c, ["apprentice_theme", "safe_archive"])
    density = ["sparse", "balanced"] if c == "story" else ["sparse", "balanced", "dense"]
    rhythm = {"story": ["minimal"], "boss": ["driving", "relentless"], "battle": ["steady", "driving", "relentless"],
              "elite": ["steady", "driving", "relentless"]}.get(c, ["minimal", "steady"])
    return {"style_id": style, "density_profile": density, "rhythm_profile": rhythm,
            "bpm_shift": [v for v, _ in MUSIC_BPM], "tension_shift": [v for v, _ in MUSIC_TENSION]}


class MusicScene:
    name = "music_scene@1"
    SYSTEM = (
        "你是卡牌战棋游戏的场景音乐导演。音乐美学是“东方Project×西方奇幻”：旋律主导，战斗要有弹幕式推进感，"
        "剧情与探索要给对白留空间；低血量、高挑战或高剧情张力时可以提高张力或速度。"
        "调式和旋律由本地导演负责，你只选编配参数，而且只能从 allowed 里给出的选项中选。"
        "只输出一个 JSON 对象，不要 Markdown：{\"style_id\":..., \"density_profile\":..., \"rhythm_profile\":..., "
        "\"bpm_shift\": 整数, \"tension_shift\": 数字}"
    )

    def key(self, row):
        return row["uid"]

    def generate(self, n, rng):
        rows = []
        for _ in range(n):
            context = rng.choices(MUSIC_CONTEXTS, weights=[1, 5, 2, 1, 4, 2, 2, 1, 1, 1])[0]
            battle = 11 if context == "boss" else rng.randint(1, 10)
            scene = {"context": context, "battle": battle, "route": rng.choice(FATE_ROUTES),
                     "challenge": rng.choice([-1, 0, 0, 1, 2, 3, 4]) if context in ("battle", "elite", "boss") else 0,
                     "hp_ratio": round(rng.uniform(0.15, 1.0), 2), "master_level": rng.choice([0, 0, 0, 1, 2])}
            if context == "story":
                scene.update(story_position=rng.choice(["opening", "pre_battle", "post_battle", "ending"]),
                             story_tension=round(rng.uniform(0.0, 1.0), 2))
            uid = f"music:{rng.getrandbits(40):x}"
            rows.append({"rec": "decision", "schema": self.name, "uid": uid, "split": _split(uid), "scene": scene})
        return rows

    def messages(self, row):
        payload = {"scene": row["scene"], "allowed": music_legal(row["scene"])}
        return [{"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]

    def parse(self, content, row):
        v = _json_obj(content)
        if not v:
            return None
        legal = music_legal(row["scene"])
        out = {}
        for f in ("style_id", "density_profile", "rhythm_profile"):
            if v.get(f) not in legal[f]:
                return None
            out[f] = v[f]
        bpm, ten = _num(v.get("bpm_shift")), _num(v.get("tension_shift"))
        if bpm is None or ten is None:
            return None
        out["bpm_shift"] = _nearest(MUSIC_BPM, bpm)
        out["tension_shift"] = _nearest(MUSIC_TENSION, ten)
        return out

    def aggregate(self, samples):
        return _field_dists(samples, MUSIC_FIELDS)


def music_s1(row):
    legal = music_legal(row["scene"])
    guide = ("游戏音乐美学是“东方Project×西方奇幻”：旋律主导、战斗有弹幕式推进感；剧情与探索要给对白留空间；"
             "低血量、高挑战或高剧情张力时可以提高张力或速度。为下面这个场景选择编配参数。")
    q = {}
    for f, texts in (("style_id", MUSIC_STYLES), ("density_profile", MUSIC_DENSITY), ("rhythm_profile", MUSIC_RHYTHM)):
        if len(legal[f]) > 1:   # single-option fields are decided by the rules, no question needed
            q[f] = {"type": "choice", "instructions": guide, "criteria": {o: texts[o] for o in legal[f]}}
    q["bpm_shift"] = {"type": "score", "instructions": guide + " 速度微调？", "criteria": [t for _, t in MUSIC_BPM]}
    q["tension_shift"] = {"type": "score", "instructions": guide + " 张力微调？", "criteria": [t for _, t in MUSIC_TENSION]}
    return row["scene"], q


def music_s1_label(ans, row):
    legal = music_legal(row["scene"])
    out = {}
    for f in ("style_id", "density_profile", "rhythm_profile"):
        out[f] = {k: float(v) for k, v in ans[f]["probabilities"].items()} if f in ans else {legal[f][0]: 1.0}
    out["bpm_shift"] = _score_dist(ans["bpm_shift"], MUSIC_BPM)
    out["tension_shift"] = _score_dist(ans["tension_shift"], MUSIC_TENSION)
    out["n_valid"] = 1
    return out


MusicScene.s1, MusicScene.s1_label = staticmethod(music_s1), staticmethod(music_s1_label)
BossTactic.s1, BossTactic.s1_label = staticmethod(boss_s1), staticmethod(boss_s1_label)
DifficultyAdaptation.s1, DifficultyAdaptation.s1_label = staticmethod(adapt_s1), staticmethod(adapt_s1_label)
NemesisDirector.s1, NemesisDirector.s1_label = staticmethod(nemesis_s1), staticmethod(nemesis_s1_label)

# ------------------------------------------------------------------ encounter_procurement@1
# Mirrors scripts/content/encounter_builder.gd. Contexts are exported from the real game
# (build_context + build_ai_request + generate_local, see local_ai_edition/tools/export_encounter_contexts.gd).
# The model scores modules per enemy slot; the game assembles the proposal and runs its own
# repair_proposal / validate_proposal (budget, counts, compatibility), falling back to generate_local.

ENC_MAX_ENEMIES = 4
ENC_CATEGORIES = ["chassis", "hp_tier", "attack", "defense", "movement", "behavior", "weakness"]
ENC_VOCAB_PATH = "out/encounter_vocab_v1.json"


def enc_allowed(row):
    """category -> {id: module dict} for this stage's allowlist."""
    return {k: {m["id"]: m for m in v if isinstance(m, dict)} for k, v in row["allowlist"].items() if isinstance(v, list)}


def enc_vocab():
    return json.load(open(ENC_VOCAB_PATH, encoding="utf-8"))


def enc_count(row):
    return int(row["context"]["contract"]["enemy_count_required"])


def enc_labels_from_enemies(enemies, row):
    """Proposal enemies -> per-slot field values, or None if ids/count are not legal for the stage."""
    allowed = enc_allowed(row)
    n = enc_count(row)
    if not isinstance(enemies, list) or len(enemies) != n:
        return None
    out = {}
    for i, e in enumerate(enemies[:ENC_MAX_ENEMIES]):
        if not isinstance(e, dict):
            return None
        for cat in ENC_CATEGORIES:
            if e.get(cat) not in allowed.get(cat, {}):
                return None
            out[f"e{i}_{cat}"] = e[cat]
        traits = e.get("traits", [])
        if not isinstance(traits, list) or any(t not in allowed.get("traits", {}) for t in traits):
            return None
        out[f"e{i}_traits"] = list(dict.fromkeys(traits))
    return out


def enc_fields(n):
    return [f"e{i}_{c}" for i in range(n) for c in ENC_CATEGORIES], [f"e{i}_traits" for i in range(n)]


class EncounterProcurement:
    name = "encounter_procurement@1"

    def key(self, row):
        return row["uid"]

    def generate(self, n, rng):
        rows, vocab = [], {c: set() for c in ENC_CATEGORIES + ["traits"]}
        for line in open("out/encounter_contexts_v1.jsonl", encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue   # a truncated trailing line if the exporter was interrupted
            for cat, mods in enc_allowed(r).items():
                if cat in vocab:
                    vocab[cat].update(mods)
            rule = enc_labels_from_enemies((r.get("local") or {}).get("proposal", {}).get("enemies"), r)
            r.update(rec="decision", schema=self.name, split=_split(r["uid"]))
            if rule:
                single, multi = enc_fields(enc_count(r))
                r["labels"] = {"rule": {"src": "rule:encounter_builder.generate_local", "n_valid": 1,
                                        **{f: {rule[f]: 1.0} for f in single},
                                        **{f: {t: 1.0 for t in rule[f]} for f in multi}}}
            rows.append(r)
            if len(rows) >= n:
                break
        json.dump({c: sorted(v) for c, v in vocab.items()}, open(ENC_VOCAB_PATH, "w", encoding="utf-8"), indent=1)
        return rows

    def messages(self, row):
        return row["messages"]   # the game's exact request (system prompt + payload)

    def parse(self, content, row):
        v = _json_obj(content)
        return enc_labels_from_enemies((v or {}).get("enemies"), row) if v else None

    def aggregate(self, samples):
        n = max(int(k[1]) for k in samples[0] if k.startswith("e")) + 1
        single, multi = enc_fields(n)
        return _field_dists(samples, single, multi=tuple(multi))


def enc_state(row):
    c = row["context"]
    snap = c.get("player_snapshot", {})
    return {"stage": row["stage"], "contract": c["contract"], "total_budget": c["budget"]["total_budget"],
            "player": {k: snap.get(k) for k in ("hp_ratio", "deck_size", "deck_tags", "choices") if k in snap}}


def enc_s1(row):
    allowed = enc_allowed(row)
    n = enc_count(row)
    contract = row["context"]["contract"]
    guide = (f"你是遭遇采购导演。本战需要恰好 {n} 个敌人，总预算 {row['context']['budget']['total_budget']}，"
             f"关卡职责：{contract.get('duty', '')}。按职责与阵型搭配机制，不要单纯堆生命；敌人之间要有分工。")
    q = {}
    for i in range(min(n, ENC_MAX_ENEMIES)):
        for cat in ENC_CATEGORIES + ["traits"]:
            mods = allowed.get(cat, {})
            if cat == "traits":
                if int(contract.get("trait_max", 0)) <= 0 or not mods:
                    continue
                crit = {"none": "不附加特性"} | {mid: f"{m.get('name', mid)}（成本{m.get('cost', 0)}）" for mid, m in mods.items()}
            else:
                if len(mods) <= 1:
                    continue
                crit = {mid: f"{m.get('name', mid)}：{m.get('effect', '')}（成本{m.get('cost', 0)}"
                             + (f"，{m['family']}" if m.get("family") else "") + "）" for mid, m in mods.items()}
            q[f"e{i}__{cat}"] = {"type": "choice", "criteria": crit,
                                 "instructions": guide + f" 第{i + 1}个敌人的 {cat} 选哪个？"}
    return enc_state(row), q


def enc_s1_label(ans, row):
    allowed = enc_allowed(row)
    n = min(enc_count(row), ENC_MAX_ENEMIES)
    out = {}
    for i in range(n):
        for cat in ENC_CATEGORIES:
            a = ans.get(f"e{i}__{cat}")
            out[f"e{i}_{cat}"] = ({k: float(v) for k, v in a["probabilities"].items()} if a
                                  else {next(iter(allowed.get(cat, {"none": 0}))): 1.0})
        a = ans.get(f"e{i}__traits")
        out[f"e{i}_traits"] = {k: float(v) for k, v in a["probabilities"].items() if k != "none"} if a else {}
    out["n_valid"] = 1
    return out


EncounterProcurement.s1, EncounterProcurement.s1_label = staticmethod(enc_s1), staticmethod(enc_s1_label)


# ------------------------------------------------------------------ card_procurement@1
# Mirrors scripts/content/card_builder.gd + scripts/content/card_effects.gd. Contexts are
# exported from the real game (export_card_contexts.gd); the model scores typed components
# per card slot (target/cost/range/tags/keywords/effect atoms), the game assembles the recipe
# from a template name/description and runs its own validate_recipe / repair_recipe, falling
# back to generate_local_batch.

CARD_MAX_CARDS = 3
CARD_MAX_EFFECTS = 5
CARD_TARGETS = ["enemy", "self", "empty", "any"]
CARD_COSTS = list(range(0, 4))
CARD_RANGES = list(range(0, 13))
CARD_VALUES = list(range(1, 15))
CARD_BUILD_TAGS = ["Fire", "Lightning", "Water", "Terrain", "Utility"]
CARD_KEYWORDS = ["exhaust", "retain", "innate", "ethereal", "once_per_battle"]
CARD_CONDITIONS = ["none", "target_burning", "target_shocked", "target_on_water", "target_on_fire",
                   "target_on_charged", "after_movement", "hp_below_half", "once_per_battle"]
CARD_POOLS = ["normal", "trait", "ultimate"]
CARD_POOL_TIERS = {"normal": [1, 2, 3], "trait": [2, 3], "ultimate": [4]}
CARD_EFFECT_LIMITS = {"normal": (1, 3), "trait": (2, 4), "ultimate": (2, 5)}
# value = (min_value, max_value); pools / min_tier / element / default target mirror card_effects.gd EFFECTS.
CARD_EFFECTS = {
    "deal_damage": (1, 12, CARD_POOLS, 1, "Neutral", "enemy"),
    "apply_burn": (1, 4, CARD_POOLS, 1, "Fire", "enemy"),
    "spread_burn": (1, 3, CARD_POOLS, 2, "Fire", "all_enemies"),
    "detonate_burn": (1, 4, CARD_POOLS, 2, "Fire", "enemy"),
    "apply_shock": (1, 4, CARD_POOLS, 1, "Lightning", "enemy"),
    "chain_shock": (1, 4, CARD_POOLS, 2, "Lightning", "enemy"),
    "detonate_shock": (1, 4, CARD_POOLS, 2, "Lightning", "enemy"),
    "create_fire_terrain": (1, 4, CARD_POOLS, 1, "Terrain", "cell"),
    "create_water_terrain": (1, 4, CARD_POOLS, 1, "Water", "cell"),
    "create_charged_terrain": (1, 4, CARD_POOLS, 2, "Lightning", "cell"),
    "react_vaporize": (1, 5, ["trait", "ultimate"], 2, "Fire", "cell"),
    "react_conduct": (1, 5, ["trait", "ultimate"], 2, "Lightning", "cell"),
    "react_overload": (1, 6, ["trait", "ultimate"], 3, "Lightning", "cell"),
    "push": (1, 3, CARD_POOLS, 1, "Utility", "enemy"),
    "pull": (1, 3, CARD_POOLS, 1, "Utility", "enemy"),
    "teleport": (1, 5, CARD_POOLS, 1, "Utility", "empty"),
    "gain_block": (2, 14, CARD_POOLS, 1, "Utility", "self"),
    "draw_cards": (1, 3, CARD_POOLS, 1, "Utility", "self"),
    "gain_mp": (1, 2, CARD_POOLS, 2, "Utility", "self"),
    "exhaust_self": (1, 1, CARD_POOLS, 1, "Neutral", "self"),
}
CARD_ALL_EFFECTS = sorted(CARD_EFFECTS)


def card_allowed(pool, tier):
    """Sorted effect_id whitelist for a (pool, tier), mirroring card_effects.gd effect_ids_for."""
    return [e for e in CARD_ALL_EFFECTS
            if pool in CARD_EFFECTS[e][2] and tier >= CARD_EFFECTS[e][3]]


def card_count(row):
    return int(row["spec"].get("count", CARD_MAX_CARDS))


def card_fields():
    single, multi = [], []
    for i in range(CARD_MAX_CARDS):
        single += [f"c{i}_target", f"c{i}_cost", f"c{i}_range"]
        for j in range(CARD_MAX_EFFECTS):
            single += [f"c{i}_fx{j}", f"c{i}_fxv{j}", f"c{i}_fxc{j}"]
        multi += [f"c{i}_tags", f"c{i}_keywords"]
    return single, multi


def card_labels_from_recipes(recipes, row):
    """Local candidates -> per-slot typed values (whitelist only)."""
    if not isinstance(recipes, list):
        return None
    n = min(len(recipes), CARD_MAX_CARDS)
    out = {}
    for i in range(n):
        r = recipes[i]
        if not isinstance(r, dict):
            return None
        out[f"c{i}_target"] = r.get("target", "enemy")
        out[f"c{i}_cost"] = int(r.get("cost", 1))
        out[f"c{i}_range"] = int(r.get("range", 3))
        effects = r.get("effects", [])
        for j in range(CARD_MAX_EFFECTS):
            if j < len(effects) and isinstance(effects[j], dict):
                e = effects[j]
                out[f"c{i}_fx{j}"] = e.get("effect_id", "none")
                out[f"c{i}_fxv{j}"] = int(e.get("value", 1))
                out[f"c{i}_fxc{j}"] = e.get("condition", "none")
            else:
                out[f"c{i}_fx{j}"] = "none"
                out[f"c{i}_fxv{j}"] = 1
                out[f"c{i}_fxc{j}"] = "none"
        out[f"c{i}_tags"] = list(r.get("tags", []))
        out[f"c{i}_keywords"] = list(r.get("keywords", []))
    return out


def card_parse_labels(obj, row):
    """Chat-teacher JSON (nested {cards:[...]}) -> validated per-slot values, or None."""
    spec = row["spec"]
    pool, tier = spec["pool"], int(spec["tier"])
    allowed = set(card_allowed(pool, tier))
    effect_max = CARD_EFFECT_LIMITS[pool][1]
    cards = obj.get("cards") if isinstance(obj, dict) else obj
    if isinstance(cards, dict):
        cards = cards.get("cards", [])
    if not isinstance(cards, list):
        return None
    n = min(card_count(row), CARD_MAX_CARDS, len(cards))
    if n == 0:
        return None
    out = {}
    for i in range(n):
        c = cards[i]
        if not isinstance(c, dict):
            return None
        t = c.get("target", "enemy")
        if t not in CARD_TARGETS:
            return None
        out[f"c{i}_target"] = t
        cost = _num(c.get("cost", 1))
        if cost is None or int(round(cost)) not in CARD_COSTS:
            return None
        out[f"c{i}_cost"] = int(round(cost))
        rng = _num(c.get("range", 3))
        if rng is None or int(round(rng)) not in CARD_RANGES:
            return None
        out[f"c{i}_range"] = int(round(rng))
        tags = c.get("tags", [])
        if not isinstance(tags, list) or any(x not in CARD_BUILD_TAGS for x in tags):
            return None
        out[f"c{i}_tags"] = list(dict.fromkeys(tags))
        kws = c.get("keywords", [])
        if not isinstance(kws, list) or any(x not in CARD_KEYWORDS for x in kws):
            return None
        out[f"c{i}_keywords"] = list(dict.fromkeys(kws))
        effects = c.get("effects", [])
        if not isinstance(effects, list):
            return None
        seen = set()
        valid_effects = []
        for e in effects[:effect_max]:
            if not isinstance(e, dict):
                return None
            fx = e.get("effect_id")
            if fx not in allowed:
                return None
            if fx in seen:
                continue
            seen.add(fx)
            v = _num(e.get("value", 1))
            if v is None or int(round(v)) < 1 or int(round(v)) > 14:
                return None
            fc = e.get("condition", "none")
            if fc not in CARD_CONDITIONS:
                return None
            valid_effects.append((fx, int(round(v)), fc))
        for j in range(CARD_MAX_EFFECTS):
            if j < len(valid_effects):
                fx, v, fc = valid_effects[j]
                out[f"c{i}_fx{j}"] = fx
                out[f"c{i}_fxv{j}"] = v
                out[f"c{i}_fxc{j}"] = fc
            else:
                out[f"c{i}_fx{j}"] = "none"
                out[f"c{i}_fxv{j}"] = 1
                out[f"c{i}_fxc{j}"] = "none"
    return out


CARD_SYSTEM = (
    "你是《虚妄余罪》的卡牌采购导演。你只能从白名单 effect 原子组合每张卡的机制，"
    "不能发明新的 effect_id、目标、条件、关键词或数值字段。文本字段（卡名/描述）由本地模板生成，"
    "你只输出类型化组件。\n"
    "只输出一个 JSON 对象：{\"cards\": [ ... ]}，cards 数组恰好包含给定数量的卡。每张卡字段：\n"
    "- pool/tier 原样填写 payload 里的值；card_id/name/description 留空字符串；\n"
    "- target 取 enemy/self/empty/any；cost 取 0..3；range 取 0..12（self 卡为 0）；\n"
    "- tags 取 build_tags 子集，必须覆盖所有效果原子的 element；keywords 取 keywords 子集（可为空）；\n"
    "- effects 是数组，每个元素 {\"effect_id\", \"value\", \"condition\"}；effect_id 取白名单 ID，"
    "value 为整数，condition 取 conditions 白名单（多数为 none）。\n"
    "约束：同一张卡每个 effect_id 最多出现一次；效果数量满足 effects_min..effects_max；"
    "0 费抽牌/回费必须带 exhaust 或 once_per_battle；地形卡 card.target 应为 any；"
    "指向型卡 range 为 1..12，self 卡 range 固定 0。\n"
    "不要 Markdown、代码围栏或额外字段。"
)


class CardProcurement:
    name = "card_procurement@1"

    def key(self, row):
        return row["uid"]

    def generate(self, n, rng):
        rows = []
        for line in open("out/card_contexts_v1.jsonl", encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rule = card_labels_from_recipes((r.get("local") or {}).get("candidates"), r)
            r.update(rec="decision", schema=self.name, split=_split(r["uid"]))
            if rule:
                single, multi = card_fields()
                r["labels"] = {"rule": {"src": "rule:card_builder.generate_local_candidates", "n_valid": 1,
                                        **{f: {rule[f]: 1.0} for f in single},
                                        **{f: {t: 1.0 for t in rule[f]} for f in multi}}}
            rows.append(r)
            if len(rows) >= n:
                break
        return rows

    def messages(self, row):
        spec = row["spec"]
        pool, tier = spec["pool"], int(spec["tier"])
        fx = {}
        for e in card_allowed(pool, tier):
            lo, hi, _, _, element, default_target = CARD_EFFECTS[e]
            fx[e] = {"value": [lo, hi], "element": element, "target": default_target}
        payload = {"spec": spec, "budget": int(row.get("budget", 0)),
                   "effect_limits": row.get("effect_limits", {}),
                   "effects": fx,
                   "card_targets": CARD_TARGETS,
                   "build_tags": CARD_BUILD_TAGS,
                   "keywords": CARD_KEYWORDS,
                   "conditions": CARD_CONDITIONS,
                   "deck_tags": row.get("deck_tags", {})}
        return [{"role": "system", "content": CARD_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]

    def parse(self, content, row):
        obj = _json_obj(content)
        return card_parse_labels(obj, row) if obj else None

    def aggregate(self, samples):
        single, multi = card_fields()
        return _field_dists(samples, single, multi=tuple(multi))


def card_state(row):
    return {"spec": row["spec"], "budget": int(row.get("budget", 0)),
            "effect_limits": row.get("effect_limits", {}), "deck_tags": row.get("deck_tags", {})}


def card_s1(row):
    spec = row["spec"]
    pool, tier = spec["pool"], int(spec["tier"])
    effect_max = int(row.get("effect_limits", {}).get("max", CARD_EFFECT_LIMITS[pool][1]))
    guide = ("你是《虚妄余罪》的卡牌采购导演，为一张卡选择类型化组件；只能从白名单取值，"
             "同卡每个 effect_id 最多一次，效果数量不超过 %d。" % effect_max)
    fx_choices = {e: f"{e}（值{e_min}..{e_max}）" for e in card_allowed(pool, tier)
                  for e_min, e_max, _, _, _, _ in [CARD_EFFECTS[e]]}
    fx_choices["none"] = "本槽不使用效果"
    q = {}
    n = min(card_count(row), CARD_MAX_CARDS)
    for i in range(n):
        q[f"c{i}__target"] = {"type": "choice", "instructions": guide + " 第%d张卡的 card.target？" % (i + 1),
                              "criteria": {"enemy": "指向敌人", "self": "只指向自身", "empty": "只指向空格", "any": "地形/格子/混合"}}
        q[f"c{i}__cost"] = {"type": "choice", "instructions": guide + " 第%d张卡的费用？" % (i + 1),
                            "criteria": {str(v): str(v) for v in CARD_COSTS}}
        q[f"c{i}__range"] = {"type": "choice", "instructions": guide + " 第%d张卡的射程？" % (i + 1),
                             "criteria": {str(v): str(v) for v in CARD_RANGES}}
        for j in range(CARD_MAX_EFFECTS):
            q[f"c{i}__fx{j}"] = {"type": "choice", "instructions": guide + " 第%d张卡效果槽%d的 effect_id？" % (i + 1, j + 1),
                                 "criteria": fx_choices}
            q[f"c{i}__fxv{j}"] = {"type": "score", "instructions": guide + " 第%d张卡效果槽%d的 value？" % (i + 1, j + 1),
                                  "criteria": [str(v) for v in CARD_VALUES]}
            q[f"c{i}__fxc{j}"] = {"type": "choice", "instructions": guide + " 第%d张卡效果槽%d的 condition？" % (i + 1, j + 1),
                                  "criteria": {c: c for c in CARD_CONDITIONS}}
    return card_state(row), q


def card_s1_label(ans, row):
    spec = row["spec"]
    pool, tier = spec["pool"], int(spec["tier"])
    allowed = set(card_allowed(pool, tier))
    out = {}
    n = min(card_count(row), CARD_MAX_CARDS)
    for i in range(n):
        for f in ("target", "cost", "range"):
            out[f"c{i}_{f}"] = {k: float(v) for k, v in ans[f"c{i}__{f}"]["probabilities"].items()}
        tags = {t: 0.0 for t in CARD_BUILD_TAGS}
        kws = {k: 0.0 for k in CARD_KEYWORDS}
        for j in range(CARD_MAX_EFFECTS):
            fx = ans[f"c{i}__fx{j}"]["probabilities"]
            out[f"c{i}_fx{j}"] = {k: float(v) for k, v in fx.items()}
            out[f"c{i}_fxv{j}"] = _score_dist(ans[f"c{i}__fxv{j}"], [(v, str(v)) for v in CARD_VALUES])
            out[f"c{i}_fxc{j}"] = {k: float(v) for k, v in ans[f"c{i}__fxc{j}"]["probabilities"].items()}
            for eid, p in fx.items():
                if eid != "none" and eid in allowed and eid in CARD_EFFECTS:
                    element = CARD_EFFECTS[eid][4]
                    if element in tags and p > 0.0:
                        tags[element] = max(tags[element], float(p))
                    if eid == "exhaust_self":
                        kws["exhaust"] = max(kws["exhaust"], float(p))
        out[f"c{i}_tags"] = {t: p for t, p in tags.items() if p > 0.0}
        out[f"c{i}_keywords"] = {k: p for k, p in kws.items() if p > 0.0}
    out["n_valid"] = 1
    return out


CardProcurement.s1, CardProcurement.s1_label = staticmethod(card_s1), staticmethod(card_s1_label)


# ------------------------------------------------------------------ music_score@1
# Mirrors scripts/sound/scene_composer.gd build_chat_request + scripts/sound/score_model.gd validate.
# Contexts are exported from the real game (export_score_contexts.gd) with the exact chat request
# (qwen teacher) and the local procedural score (rule teacher). The model composes a typed 4-bar
# grid (progression + 64 step cells); the game reconstructs [bar,step,length,degree] and runs its
# own score_model.validate, falling back to compose_local.

MUSIC_BARS = 4
MUSIC_STEPS_PER_BAR = 16
MUSIC_NOTE_LENGTHS = [1, 2, 4, 8]
MUSIC_MAX_LEAP = 9
MUSIC_PROG = list(range(0, 7))
MUSIC_STEP_TOKENS = ["rest", "hold"] + [str(d) for d in range(10)]
MUSIC_BATTLE_STYLES = ["ember_march", "storm_counterpoint", "terrain_clockwork", "nemesis_exam"]
MUSIC_BATTLE_MODES = ["minor", "harmonic_minor", "dorian", "phrygian", "melodic_minor", "mixolydian", "hungarian_minor"]
MUSIC_MODE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11], "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10], "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10], "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11], "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "hungarian_minor": [0, 2, 3, 6, 7, 8, 11],
}


def music_degree_semitones(degree, mode):
    intervals = MUSIC_MODE_INTERVALS.get(mode, MUSIC_MODE_INTERVALS["minor"])
    return intervals[degree % 7] + 12 * (degree // 7)


def music_chord_classes(root):
    return [root % 7, (root + 2) % 7, (root + 4) % 7]


def music_coerce_melody(value):
    events = []
    if not isinstance(value, list):
        return events
    for item in value:
        if isinstance(item, list) and len(item) >= 4:
            events.append([int(item[0]), int(item[1]), int(item[2]), int(item[3])])
        elif isinstance(item, dict) and "bar" in item and "step" in item:
            events.append([int(item.get("bar", 0)), int(item.get("step", 0)),
                           int(item.get("length", item.get("duration", 1))),
                           int(item.get("degree", item.get("note", 0)))])
    return events


def music_floor_note_length(length):
    for allowed in (8, 4, 2, 1):
        if allowed <= length:
            return allowed
    return 1


def music_nearest_chord_tone(degree, root):
    best, best_dist = degree, 99
    for offset in (0, 2, 4):
        base = (root + offset) % 7
        for cand in (base, base + 7):
            if 0 <= cand <= 9 and abs(cand - degree) < best_dist:
                best_dist, best = abs(cand - degree), cand
    return best


def music_repair_notes(melody):
    """Port of score_model._repair_notes: clamp degree, floor length, drop out-of-bounds."""
    result = []
    for note in melody:
        bar, step, length, degree = note
        if bar < 0 or bar >= MUSIC_BARS or step < 0 or step >= MUSIC_STEPS_PER_BAR:
            continue
        degree = max(0, min(9, degree))
        length = music_floor_note_length(min(length, MUSIC_STEPS_PER_BAR - step))
        result.append([bar, step, length, degree])
    result.sort(key=lambda n: n[0] * MUSIC_STEPS_PER_BAR + n[1])
    return result


def music_resolve_overlaps(melody):
    """Port of score_model._resolve_overlaps."""
    result = []
    for note in melody:
        if not result:
            result.append(list(note))
            continue
        prev = result[-1]
        prev_start = prev[0] * MUSIC_STEPS_PER_BAR + prev[1]
        cur_start = note[0] * MUSIC_STEPS_PER_BAR + note[1]
        if cur_start < prev_start + prev[2]:
            gap = cur_start - prev_start
            if gap <= 0:
                continue
            prev[2] = music_floor_note_length(gap)
        result.append(list(note))
    return result


def music_repair_melody_theory(melody, progression, mode):
    """Port of score_model._repair_melody_theory: drop breathing-position notes, fold wide
    leaps, snap strong beats to chord tones, drop unfixable leaps."""
    result = []
    prev_semitones = -999
    for note in melody:
        bar, step, length, degree = note
        if bar == MUSIC_BARS - 1 and step >= 14:
            continue
        if prev_semitones > -999:
            guard = 0
            while music_degree_semitones(degree, mode) - prev_semitones > MUSIC_MAX_LEAP and guard < 3:
                degree -= 7
                guard += 1
            guard = 0
            while prev_semitones - music_degree_semitones(degree, mode) > MUSIC_MAX_LEAP and guard < 3:
                degree += 7
                guard += 1
            degree = max(0, min(9, degree))
        if step % 4 == 0 and (degree % 7) not in music_chord_classes(progression[bar]):
            degree = max(0, min(9, music_nearest_chord_tone(degree, progression[bar])))
        if prev_semitones > -999 and abs(music_degree_semitones(degree, mode) - prev_semitones) > MUSIC_MAX_LEAP:
            continue
        prev_semitones = music_degree_semitones(degree, mode)
        result.append([bar, step, length, degree])
    return result


def music_validate_core(progression, melody, style_id, mode):
    """Python port of score_model.validate's progression + melody rules (harmony/bass/drums are
    deterministically added by arrange_ai_core and re-validated game-side)."""
    errors = []
    if len(progression) != MUSIC_BARS:
        return ["progression必须为%d个和弦根音级数" % MUSIC_BARS]
    distinct = set()
    for root in progression:
        if root < 0 or root > 6:
            errors.append("和弦根音级数越界：%d" % root)
        distinct.add(root)
    if int(progression[0]) != 0:
        errors.append("第1小节必须从主和弦(0)开始")
    if len(distinct) < 2:
        errors.append("和弦进行至少要有2个不同和弦")
    if progression[0:2] == progression[2:4]:
        errors.append("后2小节不得原样复制前2小节")
    prev_semitones = None
    per_bar = [0, 0, 0, 0]
    for note in melody:
        bar, step, length, degree = note
        per_bar[bar] += 1
        if bar == MUSIC_BARS - 1 and step >= 14:
            errors.append("终止小节bar%d step%d不得起音" % (bar, step))
        if step % 4 == 0 and (degree % 7) not in music_chord_classes(progression[bar]):
            errors.append("强拍音非法：bar%d step%d" % (bar, step))
        st = music_degree_semitones(degree, mode)
        if prev_semitones is not None and abs(st - prev_semitones) > MUSIC_MAX_LEAP:
            errors.append("旋律跳进非法：bar%d step%d" % (bar, step))
        prev_semitones = st
    for bar in range(MUSIC_BARS):
        if per_bar[bar] == 0:
            errors.append("第%d小节旋律为空" % (bar + 1))
    battle_style = style_id in MUSIC_BATTLE_STYLES
    min_melody = 32 if battle_style else 12
    if len(melody) < min_melody:
        errors.append("风格签名失败：至少需要%d个旋律事件" % min_melody)
    section_counts = [0, 0]
    section_notes = [[], []]
    for note in melody:
        sec = note[0] // 2
        section_counts[sec] += 1
        section_notes[sec].append([note[0] % 2, note[1], note[2], note[3]])
    min_section = 14 if battle_style else 5
    for sec in range(2):
        if section_counts[sec] < min_section:
            errors.append("第%d段至少需要%d个旋律事件" % (sec + 1, min_section))
    if section_notes[0] == section_notes[1]:
        errors.append("不能把前2小节旋律原样复制为后2小节")
    if battle_style:
        if not any(r in (5, 6) for r in progression):
            errors.append("战斗和弦进行必须至少包含一次VI或VII级")
        if mode not in MUSIC_BATTLE_MODES:
            errors.append("战斗曲mode必须为战斗调式")
    return errors


def music_melody_to_grid(melody):
    grid = ["rest"] * (MUSIC_BARS * MUSIC_STEPS_PER_BAR)
    for note in melody:
        bar, step, length, degree = note
        if 0 <= bar < MUSIC_BARS and 0 <= step < MUSIC_STEPS_PER_BAR:
            grid[bar * MUSIC_STEPS_PER_BAR + step] = str(degree)
            for k in range(1, length):
                if step + k < MUSIC_STEPS_PER_BAR:
                    grid[bar * MUSIC_STEPS_PER_BAR + step + k] = "hold"
    return grid


def music_labels_from_score(progression, melody):
    out = {}
    for b in range(MUSIC_BARS):
        out[f"prog{b}"] = progression[b] if b < len(progression) else 0
    grid = music_melody_to_grid(melody)
    for i in range(MUSIC_BARS * MUSIC_STEPS_PER_BAR):
        out[f"s{i}"] = grid[i]
    return out


def music_parse(content, row):
    v = _json_obj(content)
    if not v:
        return None
    score = v.get("score") if isinstance(v.get("score"), dict) else v
    if not isinstance(score, dict):
        return None
    prog = score.get("progression")
    if not isinstance(prog, list) or len(prog) != MUSIC_BARS:
        return None
    progression = []
    for p in prog:
        x = _num(p)
        if x is None:
            return None
        progression.append(int(round(x)))
    melody = music_coerce_melody(score.get("melody"))
    if not melody:
        return None
    style_id = row["plan"]["style_id"]
    mode = row["plan"]["mode"]
    # Mirror the game's arrange_ai_core pipeline: repair notes, resolve overlaps, then repair
    # melody theory (strong beats / leaps / breathing position) before strict validation.
    melody = music_repair_notes(melody)
    melody = music_resolve_overlaps(melody)
    melody = music_repair_melody_theory(melody, progression, mode)
    if not melody or music_validate_core(progression, melody, style_id, mode):
        return None
    return music_labels_from_score(progression, melody)


class MusicScore:
    name = "music_score@1"

    def key(self, row):
        return row["uid"]

    def generate(self, n, rng):
        rows = []
        for line in open("out/score_contexts_v1.jsonl", encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            local = r.get("local") or {}
            melody = local.get("melody", [])
            rule = music_labels_from_score(local.get("progression", [0, 0, 0, 0]), melody) if melody else None
            r.update(rec="decision", schema=self.name, split=_split(r["uid"]))
            if rule:
                r["labels"] = {"rule": {"src": "rule:score_model.compose_local", "n_valid": 1,
                                        **{f: {rule[f]: 1.0} for f in rule}}}
            rows.append(r)
            if len(rows) >= n:
                break
        return rows

    def messages(self, row):
        return row["messages"]

    def parse(self, content, row):
        return music_parse(content, row)

    def aggregate(self, samples):
        fields = [f"prog{b}" for b in range(MUSIC_BARS)] + [f"s{i}" for i in range(MUSIC_BARS * MUSIC_STEPS_PER_BAR)]
        return _field_dists(samples, fields)


def music_score_s1(row):
    scene = row["scene"]
    plan = row["plan"]
    guide = ("你是《虚妄余罪》的作曲家，写4小节×16步的A/B主题旋律。强拍（step为0/4/8/12）的degree模7必须落在当前小节和弦音内，"
             "相邻音高差不超过9个半音，第4小节step≥14不起音；战斗曲每小节8-12个旋律事件、总数至少32，两个2小节段不同。")
    q = {}
    for b in range(1, MUSIC_BARS):
        q[f"prog{b}"] = {"type": "choice", "instructions": guide + f" 第{b + 1}小节的和弦根音级数？",
                         "criteria": {str(v): str(v) for v in MUSIC_PROG}}
    for i in range(MUSIC_BARS * MUSIC_STEPS_PER_BAR):
        b, s = divmod(i, MUSIC_STEPS_PER_BAR)
        q[f"s{i}"] = {"type": "choice", "instructions": guide + f" bar{b} step{s} 这一格？",
                      "criteria": {t: t for t in MUSIC_STEP_TOKENS}}
    return {"scene": {k: scene.get(k) for k in ("context", "battle")},
            "plan": {k: plan.get(k) for k in ("style_id", "mode", "density_profile", "rhythm_profile")}}, q


def music_score_s1_label(ans, row):
    out = {"prog0": {0: 1.0}}
    for b in range(1, MUSIC_BARS):
        out[f"prog{b}"] = {int(k): float(v) for k, v in ans[f"prog{b}"]["probabilities"].items()}
    for i in range(MUSIC_BARS * MUSIC_STEPS_PER_BAR):
        out[f"s{i}"] = {k: float(v) for k, v in ans[f"s{i}"]["probabilities"].items()}
    out["n_valid"] = 1
    return out


MusicScore.s1, MusicScore.s1_label = staticmethod(music_score_s1), staticmethod(music_score_s1_label)


# ------------------------------------------------------------------ camp_assessment@1
# Mirrors scripts/camp/camp_assessment_context.gd + journey_camp_adapter._public_battle. Only the
# technical combat_assessment line is typed; the omen stays with the game's local logic. The model
# picks tempo/damage/tactics bands and an overall tier; the game assembles a Chinese line from a
# template bank (>=3 patterns per label).

CAMP_TEMPO = ["fast", "steady", "slow"]
CAMP_DAMAGE = ["high", "balanced", "low"]
CAMP_TACTICS = ["high", "balanced", "low"]
CAMP_OVERALL = ["excellent", "good", "steady", "struggling"]


def _band(v, hi, lo):
    return hi if v >= 0.7 else (lo if v <= 0.4 else "balanced")


def camp_labels_from_battle(battle):
    comp = battle.get("components", {}) if isinstance(battle.get("components"), dict) else {}
    tempo = float(comp.get("tempo", 0.5))
    survival = float(comp.get("survival", 0.5))
    tactics = float(comp.get("tactics", 0.5))
    score = float(battle.get("battle_score", 50.0))
    return {
        "tempo": "fast" if tempo >= 0.7 else ("slow" if tempo <= 0.4 else "steady"),
        "damage": "low" if survival >= 0.7 else ("high" if survival <= 0.4 else "balanced"),
        "tactics": "high" if tactics >= 0.7 else ("low" if tactics <= 0.4 else "balanced"),
        "overall": "excellent" if score >= 85 else ("good" if score >= 70 else ("steady" if score >= 50 else "struggling")),
    }


CAMP_SYSTEM = (
    "你是《虚妄余罪》营地里的木人，只生成战后技术点评的类型化标签，不写整句。"
    "依据本地给出的 battle_score（0..100）与三个分量（tempo/survival/tactics，0..1）选择标签，"
    "不得重算或虚构数值，也不得输出任何系统术语。\n"
    "只输出一个 JSON 对象：{\"tempo\": fast/steady/slow, \"damage\": high/balanced/low, "
    "\"tactics\": high/balanced/low, \"overall\": excellent/good/steady/struggling}。\n"
    "damage=low 表示承伤低（survival 高）；overall 按 battle_score 分档。不要 Markdown。"
)


def camp_synth(n, rng):
    rows = []
    grades = ["S", "A", "B", "C", "D"]
    titles = ["无名游魂", "余烬守卫", "风暴镜像", "熔炉考官", "王冠追猎者", "木人试炼", "裂隙幻影"]
    for i in range(n):
        skill = rng.betavariate(2.0, 2.0)
        score = round(min(100.0, max(0.0, skill * 95 + rng.gauss(0, 8))), 1)
        tempo = round(min(1.0, max(0.0, skill + rng.gauss(0, 0.16))), 2)
        survival = round(min(1.0, max(0.0, skill + rng.gauss(0, 0.18))), 2)
        tactics = round(min(1.0, max(0.0, skill + rng.gauss(0, 0.17))), 2)
        grade = grades[min(4, max(0, int((1.0 - skill) * 5)))]
        rows.append({
            "uid": f"camp:{rng.getrandbits(40):x}",
            "battle_index": rng.randint(1, 11),
            "battle": {"enemy_title": rng.choice(titles), "result": "victory" if rng.random() < 0.8 else "defeat",
                       "battle_score": score, "grade": grade,
                       "turn_deviation": round(rng.gauss(0, 2.5), 2),
                       "damage_deviation": round(rng.gauss(0, 8.0), 2),
                       "components": {"tempo": tempo, "survival": survival, "tactics": tactics}},
        })
    return rows


class CampAssessment:
    name = "camp_assessment@1"

    def key(self, row):
        return row["uid"]

    def generate(self, n, rng):
        rows = []
        for line in open("out/camp_contexts_v1.jsonl", encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rule = camp_labels_from_battle(r["battle"])
            r.update(rec="decision", schema=self.name, split=_split(r["uid"]))
            r["labels"] = {"rule": {"src": "rule:camp_assessment_context.fallback", "n_valid": 1,
                                    **{f: {rule[f]: 1.0} for f in rule}}}
            rows.append(r)
            if len(rows) >= n:
                break
        return rows

    def messages(self, row):
        return [{"role": "system", "content": CAMP_SYSTEM},
                {"role": "user", "content": json.dumps(row["battle"], ensure_ascii=False)}]

    def parse(self, content, row):
        obj = _json_obj(content)
        if not obj:
            return None
        out = {}
        for f, opts in (("tempo", CAMP_TEMPO), ("damage", CAMP_DAMAGE), ("tactics", CAMP_TACTICS), ("overall", CAMP_OVERALL)):
            v = obj.get(f)
            if v not in opts:
                return None
            out[f] = v
        return out

    def aggregate(self, samples):
        return _field_dists(samples, ["tempo", "damage", "tactics", "overall"])


def camp_s1(row):
    b = row["battle"]
    comp = b.get("components", {})
    guide = ("你是营地里的木人，为刚结束的一战挑技术点评标签；battle_score=%.1f，tempo=%.2f，survival=%.2f，tactics=%.2f。"
             % (float(b.get("battle_score", 50)), float(comp.get("tempo", 0.5)), float(comp.get("survival", 0.5)), float(comp.get("tactics", 0.5))))
    q = {
        "tempo": {"type": "choice", "instructions": guide + " 节奏？", "criteria": {"fast": "快", "steady": "平稳", "slow": "慢"}},
        "damage": {"type": "choice", "instructions": guide + " 承伤？", "criteria": {"high": "承伤高", "balanced": "可控", "low": "承伤低"}},
        "tactics": {"type": "choice", "instructions": guide + " 战术分量？", "criteria": {"high": "高", "balanced": "中", "low": "低"}},
        "overall": {"type": "choice", "instructions": guide + " 总体水平档位？",
                    "criteria": {"excellent": "出色", "good": "良好", "steady": "稳健", "struggling": "吃力"}},
    }
    return b, q


def camp_s1_label(ans, row):
    out = {f: {k: float(v) for k, v in ans[f]["probabilities"].items()}
           for f in ("tempo", "damage", "tactics", "overall")}
    out["n_valid"] = 1
    return out


CampAssessment.s1, CampAssessment.s1_label = staticmethod(camp_s1), staticmethod(camp_s1_label)


# ------------------------------------------------------------------ encounter_flavor@1
# Mirrors content_orchestrator.gd "flavor" category (stage_title / encounter_name / intro /
# victory_line / style_tags). Only the tone and style tags are typed; the game assembles the
# four text fields from a template bank keyed by the tone. The copy pool is collected read-only
# from %APPDATA%\\Godot\\app_userdata\\虚妄余罪\\ai_content_cache flavor/encounter candidates.

FLAVOR_TONES = ["grim", "eerie", "solemn", "tense", "triumphant", "calm"]
FLAVOR_TAGS = ["冷峻", "诡谲", "肃穆", "悲怆", "昂扬", "静谧", "压抑", "庄严", "异质", "荒诞"]


def flavor_tone(stage, kind):
    if kind == "boss":
        return "eerie" if stage >= 11 else "grim"
    if kind == "elite":
        return "solemn"
    if stage <= 3:
        return "calm"
    if stage >= 9:
        return "grim"
    return "tense"


def flavor_tags(stage, kind):
    if kind == "boss":
        return ["肃穆", "庄严", "异质"] if stage >= 11 else ["肃穆", "悲怆", "庄严"]
    if kind == "elite":
        return ["冷峻", "压抑", "庄严"]
    if stage <= 3:
        return ["静谧", "昂扬"]
    if stage >= 9:
        return ["冷峻", "压抑", "悲怆"]
    return ["冷峻", "诡谲"]


def flavor_labels_from_context(ctx):
    return {"tone": flavor_tone(int(ctx["stage"]), str(ctx["kind"])),
            "style_tags": flavor_tags(int(ctx["stage"]), str(ctx["kind"]))}


FLAVOR_SYSTEM = (
    "你是《虚妄余罪》的风味文案导演，只选择类型化标签，不写整句。"
    "依据关卡序号与关卡种类选择 tone（grim/eerie/solemn/tense/triumphant/calm）与 2 到 4 个 style_tags（中文短词）。"
    "风格标签来自白名单：冷峻/诡谲/肃穆/悲怆/昂扬/静谧/压抑/庄严/异质/荒诞。\n"
    "只输出一个 JSON 对象：{\"tone\": \"...\", \"style_tags\": [\"...\", ...]}。不要 Markdown。"
)


class EncounterFlavor:
    name = "encounter_flavor@1"

    def key(self, row):
        return row["uid"]

    def generate(self, n, rng):
        rows = []
        for line in open("out/flavor_contexts_v1.jsonl", encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rule = flavor_labels_from_context(r)
            r.update(rec="decision", schema=self.name, split=_split(r["uid"]))
            r["labels"] = {"rule": {"src": "rule:content_orchestrator.flavor_local", "n_valid": 1,
                                    "tone": {rule["tone"]: 1.0},
                                    "style_tags": {t: 1.0 for t in rule["style_tags"]}}}
            rows.append(r)
            if len(rows) >= n:
                break
        return rows

    def messages(self, row):
        return [{"role": "system", "content": FLAVOR_SYSTEM},
                {"role": "user", "content": json.dumps({"stage": row["stage"], "kind": row["kind"],
                                                        "tags": FLAVOR_TAGS, "tones": FLAVOR_TONES}, ensure_ascii=False)}]

    def parse(self, content, row):
        obj = _json_obj(content)
        if not obj:
            return None
        tone = obj.get("tone")
        if tone not in FLAVOR_TONES:
            return None
        tags = obj.get("style_tags", [])
        if not isinstance(tags, list) or not 2 <= len(tags) <= 4 or any(t not in FLAVOR_TAGS for t in tags):
            return None
        return {"tone": tone, "style_tags": list(dict.fromkeys(tags))}

    def aggregate(self, samples):
        return _field_dists(samples, ["tone"], multi=("style_tags",))


def flavor_s1(row):
    q = {"tone": {"type": "choice", "instructions": "这一关的风味基调？",
                  "criteria": {"grim": "冷峻压抑", "eerie": "诡谲异质", "solemn": "肃穆庄严",
                               "tense": "紧张推进", "triumphant": "昂扬凯旋", "calm": "静谧留白"}}}
    for t in FLAVOR_TAGS:
        q[f"tag__{t}"] = {"type": "noul", "instructions": f"这一关的风格标签是否包含「{t}」？（共选2到4个）"}
    return {"stage": int(row["stage"]), "kind": str(row["kind"])}, q


def flavor_s1_label(ans, row):
    out = {"tone": {k: float(v) for k, v in ans["tone"]["probabilities"].items()}}
    out["style_tags"] = {t: float(ans[f"tag__{t}"]["noul"]) for t in FLAVOR_TAGS}
    out["n_valid"] = 1
    return out


EncounterFlavor.s1, EncounterFlavor.s1_label = staticmethod(flavor_s1), staticmethod(flavor_s1_label)


CHANNELS = {c.name: c for c in (BossTactic(), DifficultyAdaptation(), NemesisDirector(), FateAnalysis(), MusicScene(),
                                    EncounterProcurement(), CardProcurement(), MusicScore(), CampAssessment(),
                                    EncounterFlavor())}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="generate synthetic input states for a channel")
    ap.add_argument("--channel", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = CHANNELS[a.channel].generate(a.n, random.Random(a.seed))
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} states -> {a.out}")
