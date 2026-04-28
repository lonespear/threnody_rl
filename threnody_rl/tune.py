"""Counterfactual balance tuning harness.

Take an existing checkpoint, run a baseline 5×5 win-rate matrix, then
apply unit-stat overrides (e.g. "harrow.render.ballistic_skill=4"),
re-run the same matrix, and print the per-matchup delta. No retraining
required — both sides use the same fixed policy. The delta isolates
the effect of the stat change on win rates given a fixed strategy.

Why this beats edit-and-retrain:
  - Iteration cycle: minutes instead of hours
  - Causal isolation: same policy on both sides means win-rate shifts
    come from the stat change, not from policy adaptation
  - For deeper validation, you can still retrain afterwards — but most
    "is this faction OP" questions answer cleanly without it

Override syntax (CLI):
  --override <faction>.<unit_slug>.<field>=<value>

Examples:
  python -m threnody_rl.tune \\
    --checkpoint checkpoints/threnody_step_4000000_final.pt \\
    --override harrow.render.ballistic_skill=4 \\
    --override accord.line_trooper.ranged.attacks=2 \\
    --games-per-matchup 30

Unit slugs are the unit's display name lowercased with spaces replaced
by underscores: "Line Trooper" -> line_trooper, "Render" -> render,
"Recon Strider" -> recon_strider, etc.

Field paths:
  Top-level Unit:   movement, toughness, weapon_skill, ballistic_skill,
                    wounds, save, nullfield_save, objective_control,
                    advance_bonus, can_fall_back_and_shoot
  Nested weapons:   ranged.attacks, ranged.strength, ranged.ap,
                    ranged.damage, ranged.range
                    melee.attacks,  melee.strength,  melee.ap,
                    melee.damage

Numeric values are auto-coerced (int if the original was int, else float).
Boolean values: 'true'/'false'/'1'/'0' for can_fall_back_and_shoot.
'null'/'none' clears nullfield_save.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

from threnody_rl.env import ThrenodyEnv, FACTION_KEYS, DeployStyle
from threnody_rl.env.threnody_env import ACTION_SPACE_SIZE, OBS_DIM
from threnody_rl.env.game_state import GameMode
from threnody_rl.policy import MaskedActorCritic
from threnody_rl.train import FrozenOpponent, random_opponent_act, load_checkpoint
import threnody_rl.env.threnody_env as _tenv_module


# ─── Overrides ───────────────────────────────────────────────────────────────

OverrideKey = tuple[str, str, str]   # (faction, unit_slug, field_path)


def _slug(name: str) -> str:
    """'Line Trooper' -> 'line_trooper'."""
    return re.sub(r"\s+", "_", name.strip().lower())


def _parse_value(raw: str) -> int | float | bool | None:
    """Coerce CLI string to a python value."""
    s = raw.strip()
    low = s.lower()
    if low in ("null", "none"):
        return None
    if low in ("true", "false"):
        return low == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        # Fall back to float; raises if still bad.
        return float(s)


def parse_overrides(raw_list: list[str]) -> dict[OverrideKey, object]:
    """Turn ['harrow.render.ballistic_skill=4', ...] into a dict."""
    out: dict[OverrideKey, object] = {}
    for raw in raw_list:
        if "=" not in raw:
            raise ValueError(f"Override missing '=': {raw!r}")
        path, value_s = raw.split("=", 1)
        parts = path.split(".")
        if len(parts) < 3:
            raise ValueError(
                f"Override path must be <faction>.<unit_slug>.<field>: {raw!r}"
            )
        faction = parts[0].lower()
        if faction not in FACTION_KEYS:
            raise ValueError(
                f"Unknown faction {faction!r} in override {raw!r}; "
                f"valid: {FACTION_KEYS}"
            )
        unit_slug = parts[1].lower()
        field_path = ".".join(parts[2:])
        out[(faction, unit_slug, field_path)] = _parse_value(value_s)
    return out


def _apply_field(unit, field_path: str, value: object) -> bool:
    """Set unit.<field_path> = value. Returns True on success."""
    parts = field_path.split(".")
    target = unit
    for p in parts[:-1]:
        if not hasattr(target, p):
            return False
        target = getattr(target, p)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        return False
    setattr(target, leaf, value)
    return True


def apply_overrides(units: list, overrides: dict[OverrideKey, object]) -> int:
    """Patch matching units in place. Returns count of fields patched."""
    if not overrides:
        return 0
    n = 0
    for u in units:
        slug = _slug(u.name)
        for (fac, want_slug, field), val in overrides.items():
            if u.faction == fac and slug == want_slug:
                if _apply_field(u, field, val):
                    n += 1
    return n


def install_override_hook(overrides: dict[OverrideKey, object]):
    """Monkey-patch threnody_env's local roster_for reference so units
    are patched at the moment they enter the game state. Returns a fn
    that restores the original on call."""
    original = _tenv_module.roster_for

    def patched(faction_key, team, style):
        units = original(faction_key, team, style)
        apply_overrides(units, overrides)
        return units

    _tenv_module.roster_for = patched

    def restore():
        _tenv_module.roster_for = original

    return restore


# ─── Eval (mirrors eval.py with simpler scope) ───────────────────────────────

def load_policy(path: str, device: str) -> MaskedActorCritic:
    model = MaskedActorCritic(OBS_DIM, ACTION_SPACE_SIZE).to(device)
    load_checkpoint(Path(path), model)
    model.eval()
    return model


@torch.no_grad()
def _run_match(env: ThrenodyEnv, policy: MaskedActorCritic, opponent: FrozenOpponent,
               policy_side: int, f0: str, f1: str, style: str, device: str) -> int:
    """Run a single match. Returns winner team (0/1) or -1 for draw."""
    obs, mask, info = env.reset(team0_faction=f0, team1_faction=f1, deploy_style=style)
    done = False
    while not done:
        active = info["active_team"]
        if active == policy_side:
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            m = torch.as_tensor(mask, dtype=torch.int8, device=device).unsqueeze(0)
            action, _, _ = policy.act(o, m, deterministic=False)
            a = int(action.item())
        else:
            a = opponent.act(obs, mask) if opponent is not None else random_opponent_act(mask)
        obs, _r, done, mask, info = env.step(a)
    return info.get("winner", -1)


def run_matrix(env: ThrenodyEnv, policy: MaskedActorCritic, opponent: FrozenOpponent,
               games_per_matchup: int, deploy_style: str, device: str) -> dict:
    """Run round-robin across all 25 (f0, f1) matchups. Half the games per
    matchup are played with policy on side 0, half on side 1, to remove
    side-bias. Returns dict mapping (f0, f1) -> {wins, losses, draws}."""
    results: dict[tuple[str, str], dict[str, int]] = {}
    for f0 in FACTION_KEYS:
        for f1 in FACTION_KEYS:
            tally = {"wins": 0, "losses": 0, "draws": 0}
            for i in range(games_per_matchup):
                policy_side = i % 2
                style = deploy_style if deploy_style != "random" else \
                    random.choice([DeployStyle.STANDARD, DeployStyle.FRONTAL, DeployStyle.DIAGONAL])
                winner = _run_match(env, policy, opponent, policy_side, f0, f1, style, device)
                if winner == policy_side:    tally["wins"]   += 1
                elif winner == -1:           tally["draws"]  += 1
                else:                        tally["losses"] += 1
            results[(f0, f1)] = tally
    return results


def winrate(t: dict[str, int]) -> float:
    n = t["wins"] + t["losses"] + t["draws"]
    if n == 0: return 0.0
    return t["wins"] / n


# ─── Reporting ───────────────────────────────────────────────────────────────

def _print_matrix(title: str, results: dict, label: str = "win%"):
    print(f"\n{title}")
    print("                 " + "  ".join(f"{f:>10}" for f in FACTION_KEYS))
    for f0 in FACTION_KEYS:
        row = [f"{winrate(results[(f0, f1)]):>9.1%}" for f1 in FACTION_KEYS]
        print(f"  {f0:<14} " + "  ".join(row))


def _print_delta(baseline: dict, modified: dict):
    print("\nDelta (modified − baseline)  positive = override helped player faction")
    print("                 " + "  ".join(f"{f:>10}" for f in FACTION_KEYS))
    for f0 in FACTION_KEYS:
        cells = []
        for f1 in FACTION_KEYS:
            d = winrate(modified[(f0, f1)]) - winrate(baseline[(f0, f1)])
            sign = "+" if d > 0 else ""
            cells.append(f"{sign}{d * 100:>+8.1f}%" if d != 0 else f"{0.0:>+9.1f}%")
        print(f"  {f0:<14} " + "  ".join(cells))


def _per_faction_summary(baseline: dict, modified: dict):
    print("\nPer-faction average win rate (across all 5 enemy matchups)")
    print(f"  {'faction':<14}  {'baseline':>10}  {'modified':>10}  {'delta':>10}")
    for f0 in FACTION_KEYS:
        b_avg = sum(winrate(baseline[(f0, f1)]) for f1 in FACTION_KEYS) / len(FACTION_KEYS)
        m_avg = sum(winrate(modified[(f0, f1)]) for f1 in FACTION_KEYS) / len(FACTION_KEYS)
        d = m_avg - b_avg
        print(f"  {f0:<14}  {b_avg:>9.1%}  {m_avg:>9.1%}  {d * 100:>+9.1f}%")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to a trained .pt checkpoint")
    ap.add_argument("--override", action="append", default=[],
                    help="Stat override <faction>.<unit_slug>.<field>=<value> "
                         "(may be repeated)")
    ap.add_argument("--games-per-matchup", type=int, default=20)
    ap.add_argument("--deploy-style", type=str, default="random",
                    help="standard/frontal/diagonal/random")
    ap.add_argument("--mode", type=str, default="OBJECTIVES",
                    choices=["OBJECTIVES", "DEATHMATCH"])
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for reproducibility (same seed → same matchup sequence)")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default=None,
                    help="Optional path to write a JSON dump of both matrices + delta")
    args = ap.parse_args()

    if not args.override:
        print("⚠  no --override flags supplied. Running baseline twice will give a "
              "near-zero delta (only RNG noise). Pass at least one --override to "
              "actually test a change.", file=sys.stderr)

    overrides = parse_overrides(args.override)
    if overrides:
        print("Overrides to apply:")
        for (fac, slug, field), val in overrides.items():
            print(f"  {fac}.{slug}.{field} = {val!r}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Reproducibility — same seed → same sequence of deploy styles, same RNG.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = load_policy(args.checkpoint, device)
    opponent = FrozenOpponent(policy, device)  # both sides share the same policy
    env = ThrenodyEnv(mode=GameMode.OBJECTIVES if args.mode == "OBJECTIVES" else GameMode.DEATHMATCH)

    # ─── Baseline run (no patching) ────────────────────────────────────────
    print(f"\nRunning baseline matrix ({args.games_per_matchup} games × 25 matchups = "
          f"{args.games_per_matchup * 25} games)...")
    random.seed(args.seed)  # reset for reproducible matchup sequence
    np.random.seed(args.seed)
    baseline = run_matrix(env, policy, opponent, args.games_per_matchup,
                          args.deploy_style, device)

    # ─── Modified run (overrides applied) ──────────────────────────────────
    print(f"\nRunning modified matrix with {len(overrides)} override(s)...")
    restore = install_override_hook(overrides)
    try:
        random.seed(args.seed)  # same matchup sequence as baseline → fair compare
        np.random.seed(args.seed)
        modified = run_matrix(env, policy, opponent, args.games_per_matchup,
                              args.deploy_style, device)
    finally:
        restore()

    # ─── Report ────────────────────────────────────────────────────────────
    _print_matrix("Baseline win rates (rows = player faction)", baseline)
    _print_matrix("Modified win rates", modified)
    _print_delta(baseline, modified)
    _per_faction_summary(baseline, modified)

    # ─── Persist (optional) ────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        payload = {
            "checkpoint": args.checkpoint,
            "overrides": [
                {"faction": fac, "unit": slug, "field": field, "value": val}
                for (fac, slug, field), val in overrides.items()
            ],
            "games_per_matchup": args.games_per_matchup,
            "deploy_style": args.deploy_style,
            "mode": args.mode,
            "seed": args.seed,
            "baseline": {f"{a}_vs_{b}": v for (a, b), v in baseline.items()},
            "modified": {f"{a}_vs_{b}": v for (a, b), v in modified.items()},
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote JSON dump to {out_path}")


if __name__ == "__main__":
    main()
