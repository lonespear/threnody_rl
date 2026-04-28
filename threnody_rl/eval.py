"""Evaluate a checkpoint.

Runs N matches per matchup (5 × 5 = 25 pairings) with the checkpoint
policy on one side against either (a) random opponent, (b) another
checkpoint, or (c) itself (sanity).

Usage:
  python -m threnody_rl.eval --checkpoint checkpoints/threnody_step_2000000.pt \\
                             --games-per-matchup 20 --opponent random
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

from threnody_rl.env import ThrenodyEnv, FACTION_KEYS, DeployStyle
from threnody_rl.env.threnody_env import ACTION_SPACE_SIZE, OBS_DIM
from threnody_rl.env.game_state import GameMode
from threnody_rl.policy import MaskedActorCritic
from threnody_rl.train import FrozenOpponent, random_opponent_act, load_checkpoint


def load_policy(path: str, device: str) -> MaskedActorCritic:
    model = MaskedActorCritic(OBS_DIM, ACTION_SPACE_SIZE).to(device)
    load_checkpoint(Path(path), model)
    model.eval()
    return model


@torch.no_grad()
def run_match(env: ThrenodyEnv, policy_side: int,
              policy: MaskedActorCritic,
              opponent: FrozenOpponent | None,
              f0: str, f1: str, style: str,
              device: str) -> dict:
    obs, mask, info = env.reset(team0_faction=f0, team1_faction=f1, deploy_style=style)
    done = False
    steps = 0
    while not done:
        active = info["active_team"]
        if active == policy_side:
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            m = torch.as_tensor(mask, dtype=torch.int8, device=device).unsqueeze(0)
            action, _, _ = policy.act(o, m, deterministic=False)
            a = int(action.item())
        else:
            if opponent is None:
                a = random_opponent_act(mask)
            else:
                a = opponent.act(obs, mask)
        obs, _r, done, mask, info = env.step(a)
        steps += 1
    winner = info.get("winner")
    return {
        "steps": steps,
        "winner": winner,
        "policy_won": winner == policy_side,
        "policy_lost": winner == (1 - policy_side),
        "draw": winner == -1,
        "vp": info.get("vp"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--opponent", type=str, default="random",
                    help="'random', 'self', or path to another checkpoint .pt")
    ap.add_argument("--games-per-matchup", type=int, default=20)
    ap.add_argument("--deploy-style", type=str, default="random",
                    help="standard/frontal/diagonal/random")
    ap.add_argument("--mode", type=str, default="OBJECTIVES",
                    choices=["OBJECTIVES", "DEATHMATCH"])
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    policy = load_policy(args.checkpoint, device)

    if args.opponent == "random":
        opponent = None
    elif args.opponent == "self":
        opponent = FrozenOpponent(policy, device)
    else:
        opp_model = load_policy(args.opponent, device)
        opponent = FrozenOpponent(opp_model, device)

    env = ThrenodyEnv(mode=getattr(GameMode, args.mode),
                      with_terrain=True, with_objectives=(args.mode == "OBJECTIVES"),
                      max_steps=2000, seed=args.seed)

    matchups = [(a, b) for a in FACTION_KEYS for b in FACTION_KEYS]
    results = {m: {"w": 0, "l": 0, "d": 0} for m in matchups}
    total_w = total_l = total_d = 0

    styles = ([DeployStyle.STANDARD, DeployStyle.FRONTAL, DeployStyle.DIAGONAL]
              if args.deploy_style == "random" else [args.deploy_style])

    for (f0, f1) in matchups:
        for g in range(args.games_per_matchup):
            policy_side = g % 2    # alternate sides for fairness
            style = rng.choice(styles)
            res = run_match(env, policy_side, policy, opponent, f0, f1, style, device)
            if res["policy_won"]: results[(f0, f1)]["w"] += 1; total_w += 1
            elif res["policy_lost"]: results[(f0, f1)]["l"] += 1; total_l += 1
            else: results[(f0, f1)]["d"] += 1; total_d += 1

    # Per-faction summary (policy as that faction — aggregating both sides)
    print("\n== Per-matchup (policy-as-team0 rows, opponent-as-team1 cols; cell = wins/games) ==")
    header = " " * 12 + " ".join(f"{k[:8]:>9}" for k in FACTION_KEYS)
    print(header)
    for a in FACTION_KEYS:
        row = [f"{a[:10]:<12}"]
        for b in FACTION_KEYS:
            w = results[(a, b)]["w"]; tot = sum(results[(a, b)].values())
            row.append(f"{w}/{tot:<7}" if tot else " - ")
        print(" ".join(row))

    total = total_w + total_l + total_d
    print(f"\n== Aggregate over {total} games ==")
    print(f"win%  {total_w/total:.3f}    loss% {total_l/total:.3f}    draw% {total_d/total:.3f}")


if __name__ == "__main__":
    main()
