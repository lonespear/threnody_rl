"""Recompute tune.py results under decisive-game win rate.

decisive_wr = wins / (wins + losses)        # draws excluded
total_wr    = wins / (wins + losses + draws) # tune.py default
draw_rate   = draws / (wins + losses + draws)

Why this exists: H1/H2/H4 all produced wrong-sign or absent signals when
read as aggregate-WR. The diagnosis is that defensive nerfs were
turning draws into decisives — the "buff" appearance in aggregate-WR
came from previously-draw games now resolving in the override target's
favor, not from real combat-balance shifts.

decisive-WR strips that confound. If a stat change shifts decisive-WR
by 5%+ in some direction, that's a real combat effect. If decisive-WR
barely moves while aggregate-WR moves a lot, the override was a
stalemate-vs-decisive lever, not a balance lever.

Usage:
  python -m threnody_rl.analyze_tune runs/bloom_a3.json runs/siege_range24_n90.json runs/bulwark_sv3_n90.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FACTION_KEYS = ("accord", "harrow", "ironveil", "drift", "voidborn")


def total_wr(t: dict) -> float:
    n = t["wins"] + t["losses"] + t["draws"]
    return t["wins"] / n if n else 0.0


def decisive_wr(t: dict) -> float:
    n = t["wins"] + t["losses"]
    return t["wins"] / n if n else 0.0


def draw_rate(t: dict) -> float:
    n = t["wins"] + t["losses"] + t["draws"]
    return t["draws"] / n if n else 0.0


def cell(section: dict, f0: str, f1: str) -> dict:
    return section[f"{f0}_vs_{f1}"]


def print_matrix(title: str, section: dict, metric_fn):
    print(f"\n{title}")
    print("                 " + "  ".join(f"{f:>10}" for f in FACTION_KEYS))
    for f0 in FACTION_KEYS:
        cells = "  ".join(f"{metric_fn(cell(section, f0, f1)):>9.1%}" for f1 in FACTION_KEYS)
        print(f"  {f0:<14} {cells}")


def print_delta(title: str, payload: dict, metric_fn):
    print(f"\n{title}")
    print("                 " + "  ".join(f"{f:>10}" for f in FACTION_KEYS))
    for f0 in FACTION_KEYS:
        cells = []
        for f1 in FACTION_KEYS:
            d = metric_fn(cell(payload["modified"], f0, f1)) - metric_fn(cell(payload["baseline"], f0, f1))
            cells.append(f"{d * 100:>+8.1f}%" if d != 0 else f"{0.0:>+9.1f}%")
        print(f"  {f0:<14} " + "  ".join(cells))


def per_faction_summary(payload: dict, metric_fn, label: str):
    print(f"\nPer-faction average {label} (across all 5 enemy matchups)")
    print(f"  {'faction':<14}  {'baseline':>10}  {'modified':>10}  {'delta':>10}")
    for f0 in FACTION_KEYS:
        b_avg = sum(metric_fn(cell(payload["baseline"], f0, f1)) for f1 in FACTION_KEYS) / len(FACTION_KEYS)
        m_avg = sum(metric_fn(cell(payload["modified"], f0, f1)) for f1 in FACTION_KEYS) / len(FACTION_KEYS)
        d = m_avg - b_avg
        print(f"  {f0:<14}  {b_avg:>9.1%}  {m_avg:>9.1%}  {d * 100:>+9.1f}%")


def analyze_one(path: Path):
    payload = json.loads(path.read_text())
    overrides = payload.get("overrides", [])
    n = payload.get("games_per_matchup", "?")

    print("\n" + "=" * 78)
    print(f"FILE: {path.name}")
    over_strs = [f"{o['faction']}.{o['unit']}.{o['field']}={o['value']}" for o in overrides]
    print(f"  overrides: {', '.join(over_strs) if over_strs else '(none — pure baseline)'}")
    print(f"  games/matchup: {n}, mode: {payload.get('mode', '?')}, seed: {payload.get('seed', '?')}")
    print("=" * 78)

    print_matrix("Baseline aggregate-WR (= tune.py output)", payload["baseline"], total_wr)
    print_matrix("Baseline decisive-WR (W / (W+L), draws excluded)", payload["baseline"], decisive_wr)
    print_matrix("Baseline draw rate (draws / total)", payload["baseline"], draw_rate)

    print_delta("Δ aggregate-WR (modified − baseline)", payload, total_wr)
    print_delta("Δ decisive-WR (modified − baseline)   ← TRUE balance signal", payload, decisive_wr)
    print_delta("Δ draw rate (modified − baseline)     positive = more draws", payload, draw_rate)

    per_faction_summary(payload, total_wr, "aggregate-WR")
    per_faction_summary(payload, decisive_wr, "decisive-WR")

    # Diagnostic: which cells flipped sign between aggregate and decisive WR?
    print("\nDiagnostic: cells where aggregate-WR Δ and decisive-WR Δ have different signs")
    print("(these are the cells where the aggregate-WR signal was driven by draw-rate flip, not real balance)")
    flipped = []
    for f0 in FACTION_KEYS:
        for f1 in FACTION_KEYS:
            ag = total_wr(cell(payload["modified"], f0, f1)) - total_wr(cell(payload["baseline"], f0, f1))
            dc = decisive_wr(cell(payload["modified"], f0, f1)) - decisive_wr(cell(payload["baseline"], f0, f1))
            if (ag > 0.02 and dc < -0.02) or (ag < -0.02 and dc > 0.02):
                flipped.append((f0, f1, ag, dc))
    if flipped:
        print(f"  {'cell':<25}  {'agg Δ':>10}  {'dec Δ':>10}")
        for f0, f1, ag, dc in flipped:
            print(f"  {f0:>12} vs {f1:<10}  {ag * 100:>+9.1f}%  {dc * 100:>+9.1f}%")
    else:
        print("  (none — agg and dec WR moved in the same direction in every cell)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Paths to tune.py JSON dumps")
    args = ap.parse_args()
    for p in args.paths:
        analyze_one(Path(p))


if __name__ == "__main__":
    main()
