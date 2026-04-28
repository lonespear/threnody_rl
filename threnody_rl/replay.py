"""Replay — dump a single match JSON and ASCII-render frames.

Usage:
  python -m threnody_rl.replay --checkpoint checkpoints/XXX.pt \\
         --f0 accord --f1 harrow --style standard --out replays/sample.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from threnody_rl.env import ThrenodyEnv, DeployStyle
from threnody_rl.env.threnody_env import ACTION_SPACE_SIZE, OBS_DIM
from threnody_rl.env.game_state import GameMode
from threnody_rl.policy import MaskedActorCritic
from threnody_rl.train import FrozenOpponent, random_opponent_act, load_checkpoint


def serialize_frame(env: ThrenodyEnv, info: dict, action: int | None, reward: float) -> dict:
    st = env.state
    units = []
    for u in st.units:
        units.append({
            "id": u.id, "name": u.name, "team": u.team,
            "x": round(u.x, 2), "y": round(u.y, 2),
            "hp": u.wounds_remaining, "wounds": u.wounds,
            "dead": u.is_dead, "deployed": u.deployed,
        })
    return {
        "turn": st.turn, "phase": st.phase, "active_team": st.active_team,
        "vp": list(st.vp),
        "action": action, "reward": reward,
        "units": units,
        "log_head": st.combat_log[:3],
    }


@torch.no_grad()
def run(args) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    policy = None
    opponent = None
    if args.checkpoint:
        policy = MaskedActorCritic(OBS_DIM, ACTION_SPACE_SIZE).to(device)
        load_checkpoint(Path(args.checkpoint), policy)
        policy.eval()
    if args.opponent and args.opponent != "random":
        opp = MaskedActorCritic(OBS_DIM, ACTION_SPACE_SIZE).to(device)
        load_checkpoint(Path(args.opponent), opp)
        opponent = FrozenOpponent(opp, device)

    env = ThrenodyEnv(mode=GameMode.OBJECTIVES, with_terrain=True,
                      with_objectives=True, max_steps=2000, seed=args.seed)

    obs, mask, info = env.reset(team0_faction=args.f0, team1_faction=args.f1,
                                deploy_style=args.style)
    frames = [serialize_frame(env, info, action=None, reward=0.0)]
    done = False
    steps = 0

    while not done:
        active = info["active_team"]
        if active == args.policy_side and policy is not None:
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            m = torch.as_tensor(mask, dtype=torch.int8, device=device).unsqueeze(0)
            action, _, _ = policy.act(o, m, deterministic=False)
            a = int(action.item())
        else:
            if opponent is None or active == args.policy_side:
                a = random_opponent_act(mask)
            else:
                a = opponent.act(obs, mask)
        obs, r, done, mask, info = env.step(a)
        frames.append(serialize_frame(env, info, action=a, reward=r))
        steps += 1

    out = {
        "f0": args.f0, "f1": args.f1, "style": args.style,
        "winner": info.get("winner"), "steps": steps,
        "frames": frames,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out} — winner={out['winner']} steps={steps}", flush=True)

    if args.ascii:
        print(env.render())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="policy for team0 (or whichever --policy-side)")
    ap.add_argument("--opponent", type=str, default="random",
                    help="'random' or path to another checkpoint")
    ap.add_argument("--policy-side", type=int, default=0, choices=[0, 1])
    ap.add_argument("--f0", type=str, default="accord")
    ap.add_argument("--f1", type=str, default="harrow")
    ap.add_argument("--style", type=str, default=DeployStyle.STANDARD,
                    choices=[DeployStyle.STANDARD, DeployStyle.FRONTAL, DeployStyle.DIAGONAL])
    ap.add_argument("--out", type=str, default="replays/sample.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--ascii", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
