"""PPO trainer with league self-play.

Each rollout is collected by stepping a single env where the LIVE policy
controls one team and a FROZEN snapshot controls the other. The frozen
snapshot is refreshed every `--league-interval` steps. Both teams alternate
between live and frozen across episodes so the policy gets gradient signal
on both sides.

Faction pairing each episode is random across all 25 matchups, so the
shared policy sees the full distribution.

Resume:
  python train.py --resume checkpoints/threnody_step_XXXXXXX.pt

Headless run:
  nohup python train.py --total-steps 4000000 \\
        --log-interval 5000 --save-interval 100000 \\
        > training_out.txt 2>&1 &
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from threnody_rl.env import (
    ThrenodyEnv, FACTION_KEYS, DeployStyle,
)
from threnody_rl.env.threnody_env import (
    ACTION_SPACE_SIZE, OBS_DIM,
)
from threnody_rl.env.game_state import GameMode
from threnody_rl.policy import MaskedActorCritic, HIDDEN


# ─── Hyperparameters ─────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    total_steps:    int = 4_000_000
    rollout_steps:  int = 4096            # per epoch of collection (across all envs)
    minibatch_size: int = 256
    epochs:         int = 4
    gamma:          float = 0.995
    gae_lambda:     float = 0.95
    clip_eps:       float = 0.2
    value_coef:     float = 0.5
    ent_coef:       float = 0.02
    max_grad_norm:  float = 0.5
    lr:             float = 3e-4
    league_interval: int = 100_000        # refresh frozen opponent every N global steps
    save_interval:   int = 100_000
    log_interval:    int = 5_000
    eval_episodes:   int = 50
    seed:            int = 0
    device:          str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir:  str = "checkpoints"
    log_dir:         str = "logs"


# ─── Rollout buffer ──────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, action_dim: int, device: str):
        self.size = size
        self.device = device
        self.obs       = torch.zeros((size, obs_dim), dtype=torch.float32)
        self.masks     = torch.zeros((size, action_dim), dtype=torch.int8)
        self.actions   = torch.zeros(size, dtype=torch.long)
        self.log_probs = torch.zeros(size, dtype=torch.float32)
        self.values    = torch.zeros(size, dtype=torch.float32)
        self.rewards   = torch.zeros(size, dtype=torch.float32)
        self.dones     = torch.zeros(size, dtype=torch.float32)
        self.advantages = torch.zeros(size, dtype=torch.float32)
        self.returns   = torch.zeros(size, dtype=torch.float32)
        self.idx = 0

    def add(self, obs, mask, action, log_prob, value, reward, done):
        i = self.idx
        self.obs[i] = torch.as_tensor(obs)
        self.masks[i] = torch.as_tensor(mask, dtype=torch.int8)
        self.actions[i] = int(action)
        self.log_probs[i] = float(log_prob)
        self.values[i] = float(value)
        self.rewards[i] = float(reward)
        self.dones[i] = float(done)
        self.idx += 1

    def is_full(self) -> bool:
        return self.idx >= self.size

    def reset(self):
        self.idx = 0

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        """GAE computed over only the LIVE-policy transitions stored here."""
        n = self.idx
        adv = torch.zeros(n, dtype=torch.float32)
        last_adv = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else float(self.values[t + 1])
            next_non_terminal = 1.0 - float(self.dones[t])
            delta = float(self.rewards[t]) + gamma * next_value * next_non_terminal - float(self.values[t])
            last_adv = delta + gamma * lam * next_non_terminal * last_adv
            adv[t] = last_adv
        self.advantages[:n] = adv
        self.returns[:n] = adv + self.values[:n]

    def iter_minibatches(self, batch_size: int):
        n = self.idx
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            sel = idx[start:start + batch_size]
            yield (self.obs[sel], self.masks[sel].to(torch.float32),
                   self.actions[sel], self.log_probs[sel],
                   self.advantages[sel], self.returns[sel])


# ─── League opponent pool ────────────────────────────────────────────────────

class FrozenOpponent:
    """A frozen MaskedActorCritic snapshot used as the opposing-side policy."""
    def __init__(self, model: MaskedActorCritic, device: str):
        self.model = copy.deepcopy(model).to(device).eval()
        self.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.int8, device=self.device).unsqueeze(0)
        action, _, _ = self.model.act(o, m, deterministic=False)
        return int(action.item())


def random_opponent_act(mask: np.ndarray) -> int:
    legal = np.flatnonzero(mask)
    return int(np.random.choice(legal)) if len(legal) > 0 else 0


# ─── Episode rollout ─────────────────────────────────────────────────────────

@dataclass
class EpisodeStats:
    steps: int = 0
    live_team_won: int = 0
    live_team_lost: int = 0
    draws: int = 0
    live_team_assignments: int = 0


def collect_rollout(env: ThrenodyEnv,
                    live: MaskedActorCritic,
                    frozen: FrozenOpponent | None,
                    buf: RolloutBuffer,
                    cfg: PPOConfig,
                    rng: random.Random,
                    last_obs_state: dict | None = None) -> tuple[EpisodeStats, dict]:
    """Fill the buffer to capacity, stepping the env across multiple episodes.

    Live policy generates training transitions only on its own active steps.
    Frozen opponent's actions are applied to the env but not stored in buf.
    Episode boundaries flush — we start a fresh game when one ends.
    """
    stats = EpisodeStats()
    device = cfg.device

    # State carried across collect_rollout calls so we don't waste a partial ep.
    if last_obs_state is None or last_obs_state.get("done", True):
        last_obs_state = _start_new_episode(env, rng)
        stats.live_team_assignments += 1

    obs = last_obs_state["obs"]
    mask = last_obs_state["mask"]
    info = last_obs_state["info"]
    live_team = last_obs_state["live_team"]

    while not buf.is_full():
        active_team = info["active_team"]
        if active_team == live_team:
            # LIVE policy step — store transition
            o_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            m_t = torch.as_tensor(mask, dtype=torch.int8, device=device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value = live.act(o_t, m_t)
            a_int = int(action.item())
            next_obs, reward, done, next_mask, info = env.step(a_int)
            buf.add(obs, mask, a_int, float(log_prob.item()), float(value.item()),
                    reward, float(done))
            obs, mask = next_obs, next_mask
            stats.steps += 1
        else:
            # FROZEN opponent step (no transition stored)
            if frozen is None:
                a_int = random_opponent_act(mask)
            else:
                a_int = frozen.act(obs, mask)
            next_obs, _opp_reward, done, next_mask, info = env.step(a_int)
            obs, mask = next_obs, next_mask

        if done:
            winner = info.get("winner")
            if winner == live_team: stats.live_team_won += 1
            elif winner == 1 - live_team: stats.live_team_lost += 1
            else: stats.draws += 1
            # Start new episode
            new_state = _start_new_episode(env, rng)
            obs = new_state["obs"]; mask = new_state["mask"]; info = new_state["info"]
            live_team = new_state["live_team"]
            stats.live_team_assignments += 1

    last_obs_state = {
        "obs": obs, "mask": mask, "info": info,
        "live_team": live_team, "done": False,
    }
    return stats, last_obs_state


def _start_new_episode(env: ThrenodyEnv, rng: random.Random) -> dict:
    f0 = rng.choice(FACTION_KEYS)
    f1 = rng.choice(FACTION_KEYS)
    style = rng.choice([DeployStyle.STANDARD, DeployStyle.FRONTAL, DeployStyle.DIAGONAL])
    obs, mask, info = env.reset(team0_faction=f0, team1_faction=f1, deploy_style=style)
    live_team = rng.randint(0, 1)
    return {"obs": obs, "mask": mask, "info": info, "live_team": live_team, "done": False}


# ─── PPO update ──────────────────────────────────────────────────────────────

def ppo_update(model: MaskedActorCritic, opt: optim.Optimizer, buf: RolloutBuffer,
               cfg: PPOConfig) -> dict:
    device = cfg.device
    losses = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0,
              "kl": 0.0, "clip_frac": 0.0, "n_mb": 0}

    # Normalize advantages
    n = buf.idx
    adv = buf.advantages[:n]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    buf.advantages[:n] = adv

    for _ in range(cfg.epochs):
        for batch in buf.iter_minibatches(cfg.minibatch_size):
            obs_b, mask_b, act_b, old_lp_b, adv_b, ret_b = [t.to(device) for t in batch]

            new_lp, entropy, value = model.evaluate(obs_b, mask_b, act_b)
            ratio = torch.exp(new_lp - old_lp_b)
            unclipped = ratio * adv_b
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_b
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(value, ret_b)
            ent_loss = -entropy.mean()

            loss = policy_loss + cfg.value_coef * value_loss + cfg.ent_coef * ent_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()

            with torch.no_grad():
                approx_kl = (old_lp_b - new_lp).mean().item()
                clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()
            losses["policy"] += policy_loss.item()
            losses["value"]  += value_loss.item()
            losses["entropy"] += -ent_loss.item()  # report positive entropy
            losses["total"]  += loss.item()
            losses["kl"]     += approx_kl
            losses["clip_frac"] += clip_frac
            losses["n_mb"]   += 1

    n_mb = max(1, losses["n_mb"])
    for k in ("policy", "value", "entropy", "total", "kl", "clip_frac"):
        losses[k] /= n_mb
    return losses


# ─── Checkpoint I/O ─────────────────────────────────────────────────────────

def save_checkpoint(path: Path, model: MaskedActorCritic, opt: optim.Optimizer,
                    step: int, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),   # key matches FortyK / Archon convention
        "optim_state": opt.state_dict(),
        "step": step,
        "meta": meta,
    }, path)


def load_checkpoint(path: Path, model: MaskedActorCritic,
                    opt: optim.Optimizer | None = None) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    if opt is not None and "optim_state" in ck:
        opt.load_state_dict(ck["optim_state"])
    return ck


# ─── Train loop ─────────────────────────────────────────────────────────────

def train(cfg: PPOConfig, resume: str | None = None) -> None:
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = cfg.device
    print(f"[init] device={device} obs_dim={OBS_DIM} action_dim={ACTION_SPACE_SIZE} "
          f"hidden={HIDDEN}", flush=True)

    model = MaskedActorCritic(OBS_DIM, ACTION_SPACE_SIZE).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    start_step = 0
    if resume is not None and Path(resume).exists():
        ck = load_checkpoint(Path(resume), model, opt)
        start_step = int(ck.get("step", 0))
        print(f"[resume] loaded {resume} at step {start_step}", flush=True)

    # Build reward config from CLI overrides (None = keep RewardConfig default).
    from threnody_rl.env.threnody_env import RewardConfig
    reward_cfg = RewardConfig()
    if cfg.win          is not None: reward_cfg.win = cfg.win
    if cfg.loss         is not None: reward_cfg.loss = cfg.loss
    if cfg.draw_penalty is not None: reward_cfg.draw_penalty = cfg.draw_penalty
    if cfg.dmg_dealt    is not None: reward_cfg.dmg_dealt = cfg.dmg_dealt; reward_cfg.dmg_taken = cfg.dmg_dealt
    if cfg.vp_gained    is not None: reward_cfg.vp_gained = cfg.vp_gained; reward_cfg.vp_conceded = cfg.vp_gained
    print(f"[reward] win={reward_cfg.win} loss={reward_cfg.loss} "
          f"draw_penalty={reward_cfg.draw_penalty} dmg={reward_cfg.dmg_dealt} "
          f"kill={reward_cfg.enemy_killed} vp={reward_cfg.vp_gained}", flush=True)
    print(f"[mode] {cfg.mode}", flush=True)

    mode = GameMode.DEATHMATCH if cfg.mode == "DEATHMATCH" else GameMode.OBJECTIVES
    env = ThrenodyEnv(mode=mode, with_terrain=True,
                      with_objectives=(cfg.mode == "OBJECTIVES"),
                      max_steps=2000, seed=cfg.seed,
                      reward_cfg=reward_cfg)
    buf = RolloutBuffer(cfg.rollout_steps, OBS_DIM, ACTION_SPACE_SIZE, device)

    # Initial frozen opponent: a copy of the live policy at start.
    frozen = FrozenOpponent(model, device)
    last_freeze_step = start_step
    print(f"[league] frozen opponent initialized at step {start_step}", flush=True)

    last_obs_state = None
    rolling_winrate = deque(maxlen=200)
    rolling_lossrate = deque(maxlen=200)
    rolling_drawrate = deque(maxlen=200)
    rolling_eplen   = deque(maxlen=200)
    last_log_step = start_step
    last_save_step = start_step
    t0 = time.time()
    global_step = start_step

    while global_step < cfg.total_steps:
        buf.reset()
        ep_stats, last_obs_state = collect_rollout(env, model, frozen, buf, cfg,
                                                   rng, last_obs_state)
        global_step += buf.idx

        # Bootstrap value for GAE
        with torch.no_grad():
            o_t = torch.as_tensor(last_obs_state["obs"], dtype=torch.float32,
                                  device=device).unsqueeze(0)
            _, last_value = model(o_t)
            last_value = float(last_value.item())
        # If the last collected step ended an episode, the bootstrap should be 0
        if buf.idx > 0 and float(buf.dones[buf.idx - 1]) > 0.5:
            last_value = 0.0
        buf.compute_gae(last_value, cfg.gamma, cfg.gae_lambda)

        losses = ppo_update(model, opt, buf, cfg)

        # Track rolling win/loss/draw rate of LIVE policy
        n_eps = ep_stats.live_team_won + ep_stats.live_team_lost + ep_stats.draws
        if n_eps > 0:
            rolling_winrate.append(ep_stats.live_team_won  / n_eps)
            rolling_lossrate.append(ep_stats.live_team_lost / n_eps)
            rolling_drawrate.append(ep_stats.draws          / n_eps)
            rolling_eplen.append(buf.idx / max(1, n_eps))

        # League refresh
        if global_step - last_freeze_step >= cfg.league_interval:
            frozen = FrozenOpponent(model, device)
            last_freeze_step = global_step
            print(f"[league] frozen opponent refreshed at step {global_step}", flush=True)

        # Logging
        if global_step - last_log_step >= cfg.log_interval:
            elapsed = time.time() - t0
            fps = (global_step - start_step) / max(1.0, elapsed)
            def _avg(buf): return (sum(buf) / len(buf)) if buf else float("nan")
            wr_avg = _avg(rolling_winrate)
            lr_avg = _avg(rolling_lossrate)
            dr_avg = _avg(rolling_drawrate)
            ep_len_avg = _avg(rolling_eplen)
            # decisive = wins + losses (anything but draws). Useful as the
            # canary for the Archon-style "policy converges to drawing"
            # failure mode — if dec_avg drifts down, raise draw_penalty.
            dec_avg = (wr_avg + lr_avg) if not (math.isnan(wr_avg) or math.isnan(lr_avg)) else float("nan")
            print(
                f"[step {global_step:>9d}] wr={wr_avg:.3f} lr={lr_avg:.3f} dr={dr_avg:.3f} dec={dec_avg:.3f} "
                f"ep_len={ep_len_avg:.1f} "
                f"pol_loss={losses['policy']:+.4f} val_loss={losses['value']:.4f} "
                f"ent={losses['entropy']:.3f} kl={losses['kl']:+.4f} "
                f"clip_frac={losses['clip_frac']:.3f} fps={fps:.0f} "
                f"buf={buf.idx}",
                flush=True,
            )
            last_log_step = global_step

        # Checkpoint
        if global_step - last_save_step >= cfg.save_interval:
            ck_path = Path(cfg.checkpoint_dir) / f"threnody_step_{global_step:07d}.pt"
            save_checkpoint(ck_path, model, opt, global_step, meta={
                "obs_dim": OBS_DIM, "action_dim": ACTION_SPACE_SIZE,
                "hidden": HIDDEN, "total_steps_target": cfg.total_steps,
            })
            print(f"[save] {ck_path}", flush=True)
            last_save_step = global_step

    # Final save
    ck_path = Path(cfg.checkpoint_dir) / f"threnody_step_{global_step:07d}_final.pt"
    save_checkpoint(ck_path, model, opt, global_step, meta={"final": True})
    print(f"[done] final checkpoint -> {ck_path}", flush=True)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=4_000_000)
    ap.add_argument("--rollout-steps", type=int, default=4096)
    ap.add_argument("--minibatch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--league-interval", type=int, default=100_000)
    ap.add_argument("--save-interval", type=int, default=100_000)
    ap.add_argument("--log-interval", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    ap.add_argument("--log-dir", type=str, default="logs")
    ap.add_argument("--resume", type=str, default=None)
    # Reward / mode tuning — escape the draw plateau without recompiling.
    ap.add_argument("--mode", type=str, default="DEATHMATCH",
                    choices=["DEATHMATCH", "OBJECTIVES"],
                    help="Game mode for training episodes. DEATHMATCH ties "
                         "are rarer than OBJECTIVES VP ties → faster decisive "
                         "learning. Eval can still be run on either mode.")
    ap.add_argument("--win",           type=float, default=None,
                    help="Override RewardConfig.win (default 50.0)")
    ap.add_argument("--loss",          type=float, default=None,
                    help="Override RewardConfig.loss (default 50.0)")
    ap.add_argument("--draw-penalty",  type=float, default=None,
                    help="Override RewardConfig.draw_penalty (default 20.0)")
    ap.add_argument("--dmg-dealt",     type=float, default=None,
                    help="Override RewardConfig.dmg_dealt shaping coefficient")
    ap.add_argument("--vp-gained",     type=float, default=None,
                    help="Override RewardConfig.vp_gained shaping coefficient")
    args = ap.parse_args()

    cfg = PPOConfig(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,
        value_coef=args.value_coef,
        league_interval=args.league_interval,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        seed=args.seed,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
