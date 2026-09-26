"""Canonical renderers extracted from the training pipeline. Apache-2.0."""
import math

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
