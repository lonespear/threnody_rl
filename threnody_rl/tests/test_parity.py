"""Parity and sanity tests for the Python engine port.

Run:
  python -m threnody_rl.tests.test_parity
"""

from __future__ import annotations

import math
import random
import sys
import traceback

import numpy as np

from threnody_rl.env.unit import (
    Unit, Weapon, wound_target,
    expected_damage_ranged, expected_damage_melee,
)
from threnody_rl.env.board import Board
from threnody_rl.env.factions import FACTION_KEYS, roster_for, DeployStyle
from threnody_rl.env.game_state import GameState, Phase, GameMode, default_objectives
from threnody_rl.env.board import default_terrain_portrait_44x60
from threnody_rl.env import ThrenodyEnv
from threnody_rl.env.threnody_env import ACTION_SPACE_SIZE


# ─── Wound-target table (reference: 10e Warhammer style used in JS) ──────────

def test_wound_target_table():
    # Direct cases from the JS branching
    assert wound_target(10, 3) == 2   # S >= 2T
    assert wound_target(5, 3)  == 3   # S > T
    assert wound_target(4, 4)  == 4   # equal
    assert wound_target(3, 5)  == 5   # T > S
    assert wound_target(3, 6)  == 6   # T >= 2S
    print("[ok] wound_target table")


# ─── Closed-form vs empirical damage distribution ────────────────────────────

def _make_unit(name, bs, ws, t, sv, nf, r_s, r_ap, r_dmg, r_att, r_kw=None,
               m_s=3, m_ap=0, m_dmg=1, m_att=1):
    u = Unit(
        id=name, name=name, faction="accord", team=0,
        x=0, y=0, movement=6, toughness=t, weapon_skill=ws,
        ballistic_skill=bs, wounds=999, save=sv, nullfield_save=nf,
        ranged=Weapon("rng", r_att, r_s, r_ap, r_dmg, range=24, keywords=r_kw or []),
        melee=Weapon("mel", m_att, m_s, m_ap, m_dmg),
    )
    u.wounds_remaining = 999
    return u


def test_empirical_damage_matches_closed_form():
    """Shoot 20000 times, compare empirical mean damage to closed-form."""
    rng = random.Random(42)
    attacker = _make_unit("atk", bs=3, ws=3, t=4, sv=3, nf=None,
                          r_s=5, r_ap=-1, r_dmg=1, r_att=2, r_kw=["Volley Fire 1"])
    target   = _make_unit("tgt", bs=4, ws=4, t=3, sv=4, nf=None,
                          r_s=3, r_ap=0, r_dmg=1, r_att=1)

    N = 20000
    total = 0
    for _ in range(N):
        target.wounds_remaining = 999
        target.is_dead = False
        attacker.has_shot = False
        res = attacker.shoot(target, distance_to=8.0, rng=rng)  # half range → VF bonus
        total += res["damage"]
    empirical = total / N
    closed = expected_damage_ranged(attacker, target, distance=8.0)
    # Tolerance: closed-form mean ± 3σ of sample mean. Std-dev is loose upper-bounded
    # by closed_form itself; ~0.05 abs tolerance is safe at N=20k.
    tol = max(0.05, 0.10 * closed)
    assert abs(empirical - closed) < tol, (
        f"ranged damage drift: empirical={empirical:.4f} closed={closed:.4f} tol={tol:.4f}"
    )
    print(f"[ok] ranged damage parity: empirical={empirical:.3f} closed={closed:.3f}")

    # Melee
    N = 20000
    total = 0
    for _ in range(N):
        target.wounds_remaining = 999
        target.is_dead = False
        attacker.has_fought = False
        res = attacker.fight(target, rng=rng)
        total += res["damage"]
    empirical = total / N
    closed = expected_damage_melee(attacker, target)
    tol = max(0.05, 0.10 * closed)
    assert abs(empirical - closed) < tol, (
        f"melee damage drift: empirical={empirical:.4f} closed={closed:.4f} tol={tol:.4f}"
    )
    print(f"[ok] melee damage parity: empirical={empirical:.3f} closed={closed:.3f}")


# ─── Board LOS + Dijkstra sanity ─────────────────────────────────────────────

def test_los_blocked_by_wall():
    bd = Board(60, 44, 18)
    bd.add_terrain({"x": 20, "y": 10, "w": 10, "h": 4})
    assert bd.has_los(15, 12, 35, 12) is False  # through the wall
    assert bd.has_los(15, 5,  35, 5)  is True   # around it
    print("[ok] LOS blocked by wall")


def test_dijkstra_respects_blocked_terrain():
    bd = Board(60, 44, 18)
    # Put a wall blocking direct movement between (5, 5) and (5, 20)
    bd.add_terrain({"x": 0, "y": 12, "w": 40, "h": 2})
    grid = bd.flood_fill(5, 5, 20, is_infantry=True)
    # Destination straight through the wall should be reachable only by going around
    dst_c = int(round(5 / 0.5)); dst_r = int(round(20 / 0.5))
    direct_cost = grid[dst_r * bd.nav_cols + dst_c]
    # Remove wall — should be faster
    bd2 = Board(60, 44, 18)
    grid2 = bd2.flood_fill(5, 5, 20, is_infantry=True)
    direct2 = grid2[dst_r * bd2.nav_cols + dst_c]
    assert direct2 < direct_cost, "pathfinding around wall should be more expensive than free path"
    print(f"[ok] Dijkstra: free cost {direct2:.1f}\" vs. walled cost {direct_cost:.1f}\"")


# ─── Faction rosters ────────────────────────────────────────────────────────

def test_all_factions_instantiate():
    for k in FACTION_KEYS:
        units = roster_for(k, team=0, style=DeployStyle.STANDARD)
        assert len(units) > 0, f"faction {k} has empty roster"
        for u in units:
            assert u.wounds_remaining == u.wounds
            assert u.team == 0
    print(f"[ok] all {len(FACTION_KEYS)} factions instantiate cleanly")


def test_faction_total_wounds_counted():
    """Sanity check total wounds per faction — counted directly from the
    JS rosters (drift.js header says 14 but the actual roster is 12:
    5*1 + 2*2 + 1*3 = 12; similar rechecks below)."""
    expected = {
        "accord":   4*1 + 1*2 + 1*4,           # 4 line troopers + shield warden + recon = 10
        "harrow":   4*2 + 1*5 + 1*2,           # 4 render + breaker + phantom = 15
        "ironveil": 3*3 + 1*5 + 1*6,           # 3 bulwark + siege + aegis = 20
        "drift":    5*1 + 2*2 + 1*3,           # 5 umbra + 2 skulker + bloom = 12
        "voidborn": 3*2 + 1*4 + 1*6,           # 3 drudge + siphon + magister = 16
    }
    for k, target in expected.items():
        roster = roster_for(k, team=0)
        total = sum(u.wounds for u in roster)
        assert total == target, f"{k} total wounds {total} != expected {target}"
    print(f"[ok] faction wound totals counted correctly")


# ─── Deployment / deploy-zone tests ──────────────────────────────────────────

def test_deploy_zones_dont_overlap():
    from threnody_rl.env.game_state import deploy_zone
    for style in ("standard", "frontal", "diagonal"):
        z0 = deploy_zone(0, style); z1 = deploy_zone(1, style)
        # no overlap
        x_overlap = not (z0["x"] + z0["w"] <= z1["x"] or z1["x"] + z1["w"] <= z0["x"])
        y_overlap = not (z0["y"] + z0["h"] <= z1["y"] or z1["y"] + z1["h"] <= z0["y"])
        assert not (x_overlap and y_overlap), f"zones overlap for style {style}"
    print("[ok] deployment zones are disjoint per style")


# ─── End-to-end smoke test: random-policy games complete ─────────────────────

def test_random_policy_games_complete(n_games: int = 50):
    env = ThrenodyEnv(mode=GameMode.OBJECTIVES, with_terrain=True,
                      with_objectives=True, max_steps=2000, seed=123)
    errors = 0
    wins = [0, 0, 0]  # team0, team1, draw
    for g in range(n_games):
        f0 = random.choice(FACTION_KEYS)
        f1 = random.choice(FACTION_KEYS)
        style = random.choice([DeployStyle.STANDARD, DeployStyle.FRONTAL, DeployStyle.DIAGONAL])
        try:
            obs, mask, info = env.reset(team0_faction=f0, team1_faction=f1, deploy_style=style)
            done = False
            while not done:
                legal = np.flatnonzero(mask)
                if len(legal) == 0:
                    # No legal actions — env should never hit this if masks are correct
                    raise RuntimeError("no legal actions available")
                a = int(np.random.choice(legal))
                obs, r, done, mask, info = env.step(a)
            w = info["winner"]
            if w == 0: wins[0] += 1
            elif w == 1: wins[1] += 1
            else: wins[2] += 1
        except Exception as e:
            errors += 1
            print(f"  game {g} [{f0} vs {f1} / {style}] errored: {e}", file=sys.stderr)
            traceback.print_exc()

    assert errors == 0, f"{errors}/{n_games} games failed"
    print(f"[ok] {n_games} random-policy games: {wins[0]} t0 wins, "
          f"{wins[1]} t1 wins, {wins[2]} draws")


# ─── Action mask sanity ──────────────────────────────────────────────────────

def test_action_mask_always_has_legal_action():
    env = ThrenodyEnv(mode=GameMode.OBJECTIVES, seed=99)
    for trial in range(10):
        obs, mask, info = env.reset()
        for _ in range(200):
            assert mask.sum() > 0, (f"empty mask on turn {env.state.turn} "
                                    f"phase {env.state.phase}")
            legal = np.flatnonzero(mask)
            a = int(np.random.choice(legal))
            obs, r, done, mask, info = env.step(a)
            if done:
                break
    print("[ok] action mask always has at least one legal action mid-game")


# ─── Driver ─────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_wound_target_table,
        test_empirical_damage_matches_closed_form,
        test_los_blocked_by_wall,
        test_dijkstra_respects_blocked_terrain,
        test_all_factions_instantiate,
        test_faction_total_wounds_counted,
        test_deploy_zones_dont_overlap,
        test_action_mask_always_has_legal_action,
        test_random_policy_games_complete,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {t.__name__}: {e}", file=sys.stderr)
        except Exception as e:
            failures += 1
            print(f"[ERROR] {t.__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
    print()
    if failures:
        print(f"*** {failures}/{len(tests)} tests failed ***", file=sys.stderr)
        sys.exit(1)
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    main()
