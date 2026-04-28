"""ThrenodyEnv — headless Gym-style env for PPO self-play.

Observation:
  Always from the ACTIVE team's perspective. Team 0 in the observation is
  "me"; team 1 is "them". The caller does NOT swap perspective — the env
  flips units/factions/vp internally on active_team change.

Action space:
  Single flat Discrete, masked per-phase. Layout:
    0                           : end-phase (advance to next phase)
    1..8                        : skip unit i's remaining activation this phase
    9..9  + 8*25 - 1            : (unit, move_offset) — MANEUVER
    209..209 + 8*16 - 1         : (unit, deploy_slot) — DEPLOY
    337..337 + 8*8  - 1         : (unit, enemy_target) — RANGED
    401..401 + 8*8  - 1         : (unit, enemy_target) — ENGAGE
    465..465 + 8*8  - 1         : (unit, enemy_target) — MELEE
  Total = 529.

Move offsets: 5×5 grid of (dx, dy) ∈ {-1,-0.5,0,+0.5,+1}^2 scaled by unit.movement.
Deploy slots: 4×4 grid over the team's deployment zone.

Reward (active team, per step):
  + 0.10 * damage dealt this step
  - 0.10 * damage taken this step
  + 2.00 * enemies killed this step
  - 2.00 * own units killed this step
  + 0.50 * VP gained this step  (Objectives mode)
  - 0.50 * VP conceded this step
  Terminal: ±10 on win/loss (+0 on draw), applied to the side whose step
  caused game_over; the opposing side's next observation receives the
  mirrored terminal reward through the "opponent last delta" accumulator.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from .board import Board, Terrain, default_terrain_portrait_44x60
from .unit import Unit
from .game_state import (
    GameState, Phase, GameMode, MAX_TURNS,
    deploy_zone, contact_range, default_objectives,
)
from .factions import (
    FACTIONS, FACTION_KEYS, DeployStyle, roster_for,
)


# ─── Sizing constants ────────────────────────────────────────────────────────

MAX_UNITS_PER_TEAM = 8
NUM_FACTIONS       = 5
NUM_MOVE_OFFSETS   = 25         # 5x5 grid
NUM_DEPLOY_SLOTS   = 16         # 4x4 grid in zone
NUM_TARGETS        = MAX_UNITS_PER_TEAM
NUM_PHASES         = 5          # DEPLOY, MANEUVER, RANGED, ENGAGE, MELEE
NUM_DEPLOY_STYLES  = 3
NUM_OBJECTIVES     = 3

# Action index ranges (inclusive-exclusive) — encoded once to keep step()
# and mask-building in sync.
A_END_PHASE = 0
A_SKIP_UNIT_START = 1
A_SKIP_UNIT_END   = A_SKIP_UNIT_START + MAX_UNITS_PER_TEAM   # 9

A_MOVE_START      = A_SKIP_UNIT_END                          # 9
A_MOVE_END        = A_MOVE_START + MAX_UNITS_PER_TEAM * NUM_MOVE_OFFSETS  # 209

A_DEPLOY_START    = A_MOVE_END                               # 209
A_DEPLOY_END      = A_DEPLOY_START + MAX_UNITS_PER_TEAM * NUM_DEPLOY_SLOTS  # 337

A_RANGED_START    = A_DEPLOY_END                             # 337
A_RANGED_END      = A_RANGED_START + MAX_UNITS_PER_TEAM * NUM_TARGETS  # 401

A_ENGAGE_START    = A_RANGED_END                             # 401
A_ENGAGE_END      = A_ENGAGE_START + MAX_UNITS_PER_TEAM * NUM_TARGETS  # 465

A_MELEE_START     = A_ENGAGE_END                             # 465
A_MELEE_END       = A_MELEE_START + MAX_UNITS_PER_TEAM * NUM_TARGETS   # 529

ACTION_SPACE_SIZE = A_MELEE_END   # 529

# Move offset grid — 5×5 from -1 to +1 in both axes, scaled by unit.movement.
_MOVE_GRID = np.array(
    [(dx, dy)
     for dy in (-1.0, -0.5, 0.0, 0.5, 1.0)
     for dx in (-1.0, -0.5, 0.0, 0.5, 1.0)],
    dtype=np.float32,
)
assert _MOVE_GRID.shape == (NUM_MOVE_OFFSETS, 2)

# Deploy slot grid — 4×4 over the zone as fractional offsets.
_DEPLOY_GRID = np.array(
    [(fx, fy)
     for fy in (0.15, 0.40, 0.60, 0.85)
     for fx in (0.15, 0.40, 0.60, 0.85)],
    dtype=np.float32,
)
assert _DEPLOY_GRID.shape == (NUM_DEPLOY_SLOTS, 2)

# ─── Per-unit feature layout ─────────────────────────────────────────────────
# 30 scalar features per unit slot.
UNIT_FEATURES = 30
OBS_GLOBALS = (
    NUM_PHASES          # phase one-hot
    + 1                 # turn normalized
    + 1                 # active_team (always 0 from agent perspective, kept for shape)
    + 2                 # VP me/them
    + 1                 # game_mode flag (0=DM, 1=Obj)
    + NUM_FACTIONS      # my faction one-hot
    + NUM_FACTIONS      # their faction one-hot
    + NUM_DEPLOY_STYLES # deploy style one-hot
)
OBS_OBJECTIVES = NUM_OBJECTIVES * 5   # x, y, ctrl_me, ctrl_them, ctrl_none
OBS_UNITS      = 2 * MAX_UNITS_PER_TEAM * UNIT_FEATURES
OBS_DIM        = OBS_GLOBALS + OBS_OBJECTIVES + OBS_UNITS
# ~520 dims. Kept small so PPO MLP is cheap.


# ─── Normalization constants ─────────────────────────────────────────────────

BOARD_W = 60.0
BOARD_H = 44.0

NORM_MOVEMENT   = 10.0
NORM_TOUGH      = 7.0
NORM_WS         = 5.0
NORM_BS         = 5.0
NORM_WOUNDS     = 6.0
NORM_SAVE       = 6.0
NORM_OBJ_CTRL   = 3.0
NORM_RANGE      = 36.0
NORM_ATTACKS    = 5.0
NORM_STRENGTH   = 8.0
NORM_DAMAGE     = 3.0
NORM_AP         = 3.0


# ─── Reward shaping ─────────────────────────────────────────────────────────

@dataclass
class RewardConfig:
    dmg_dealt: float = 0.10
    dmg_taken: float = 0.10
    enemy_killed: float = 2.0
    own_killed: float = 2.0
    vp_gained: float = 0.50
    vp_conceded: float = 0.50
    win: float = 10.0
    loss: float = 10.0


# ─── Env ─────────────────────────────────────────────────────────────────────

class ThrenodyEnv:
    """Single-agent Gym-style env; step()'s perspective flips with active_team.

    Training loop should dispatch between live and frozen policies based on
    info['active_team'] returned in the latest step/reset. Both policies
    share this one env instance.
    """

    metadata = {"render.modes": ["ascii"]}

    def __init__(self,
                 team0_faction: str | None = None,
                 team1_faction: str | None = None,
                 deploy_style: str = DeployStyle.STANDARD,
                 mode: str = GameMode.OBJECTIVES,
                 with_terrain: bool = True,
                 with_objectives: bool = True,
                 reward_cfg: RewardConfig | None = None,
                 max_steps: int = 2000,
                 seed: int | None = None):
        self._team0_faction = team0_faction
        self._team1_faction = team1_faction
        self._deploy_style = deploy_style
        self._mode = mode
        self._with_terrain = with_terrain
        self._with_objectives = with_objectives
        self.reward_cfg = reward_cfg or RewardConfig()
        self.max_steps = max_steps

        self.action_space_size = ACTION_SPACE_SIZE
        self.obs_dim = OBS_DIM

        self._rng = random.Random(seed) if seed is not None else random.Random()

        # Set after reset()
        self.board: Board | None = None
        self.state: GameState | None = None
        self.team_factions: list[str] = []
        self.step_count: int = 0

        # Cross-team damage / kill / VP accounting for reward shaping.
        # Indexed by [team].
        self._cum_dmg_dealt = [0, 0]
        self._cum_dmg_taken = [0, 0]
        self._cum_kills     = [0, 0]
        self._cum_losses    = [0, 0]
        self._cum_vp        = [0, 0]

        # Per-team snapshot at the most recent time THIS TEAM last observed
        # (so reward = delta since that snapshot). We ship rewards to the
        # team currently being served by step()'s observation.
        self._snap = [self._empty_snap(), self._empty_snap()]

        # Remember the slot position of each unit (stable index 0..7) so
        # the action decoder can map (unit_i) → Unit consistently.
        self._team_units: list[list[Unit]] = [[], []]

    # ─── Lifecycle ─────────────────────────────────────────────────────────

    def reset(self, team0_faction: str | None = None,
              team1_faction: str | None = None,
              deploy_style: str | None = None,
              seed: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        if seed is not None:
            self._rng = random.Random(seed)

        f0 = team0_faction or self._team0_faction or self._rng.choice(FACTION_KEYS)
        f1 = team1_faction or self._team1_faction or self._rng.choice(FACTION_KEYS)
        style = deploy_style or self._deploy_style
        self.team_factions = [f0, f1]

        self.board = Board(BOARD_W, BOARD_H, px_per_in=18.0)
        if self._with_terrain:
            for t in default_terrain_portrait_44x60():
                self.board.add_terrain(t)

        units0 = roster_for(f0, team=0, style=style)
        units1 = roster_for(f1, team=1, style=style)
        all_units = units0 + units1
        self._team_units = [units0, units1]

        objectives = default_objectives() if (self._with_objectives and
                                              self._mode == GameMode.OBJECTIVES) else []

        self.state = GameState(
            units=all_units,
            objectives=objectives,
            mode=self._mode,
            deploy_style=style,
            rng=self._rng,
        )
        self.step_count = 0

        self._cum_dmg_dealt = [0, 0]
        self._cum_dmg_taken = [0, 0]
        self._cum_kills     = [0, 0]
        self._cum_losses    = [0, 0]
        self._cum_vp        = [0, 0]
        self._snap = [self._empty_snap(), self._empty_snap()]

        obs = self._build_obs()
        mask = self._build_mask()
        info = {
            "active_team": self.state.active_team,
            "phase": self.state.phase,
            "turn": self.state.turn,
            "team_factions": tuple(self.team_factions),
        }
        return obs, mask, info

    # ─── Step ──────────────────────────────────────────────────────────────

    def step(self, action: int) -> tuple[np.ndarray, float, bool, np.ndarray, dict]:
        if self.state is None or self.board is None:
            raise RuntimeError("call reset() before step()")

        acting_team = self.state.active_team
        mask = self._build_mask()

        legal = bool(mask[action])
        if not legal:
            # Illegal action: treat as end-phase (defensive). Trainer should
            # be using masks so this path is rare.
            action = A_END_PHASE

        self._apply_action(action)
        self.step_count += 1

        # After applying, phase / active_team may have changed. Game may end.
        done = self.state.game_over or self.step_count >= self.max_steps
        if self.step_count >= self.max_steps and not self.state.game_over:
            self.state._resolve_game_end()
            done = True

        # Compute reward to deliver to whichever team's observation is served
        # next (could still be `acting_team` mid-phase, or the other team).
        next_team = self.state.active_team
        obs = self._build_obs()
        new_mask = self._build_mask()

        # Reward for the team *receiving* the next observation
        reward = self._pop_reward(next_team, done=done)

        info = {
            "active_team": next_team,
            "acting_team": acting_team,
            "phase": self.state.phase,
            "turn": self.state.turn,
            "vp": tuple(self.state.vp),
            "team_factions": tuple(self.team_factions),
            "game_over": self.state.game_over,
            "winner": self.state.winner,
            "step_count": self.step_count,
        }
        return obs, reward, done, new_mask, info

    # ─── Action application ────────────────────────────────────────────────

    def _apply_action(self, action: int) -> None:
        st = self.state
        bd = self.board
        assert st is not None and bd is not None

        # End phase
        if action == A_END_PHASE:
            self._end_phase()
            return

        # Skip unit (cosmetic — mark unit's phase-specific flag so mask drops it)
        if A_SKIP_UNIT_START <= action < A_SKIP_UNIT_END:
            u_idx = action - A_SKIP_UNIT_START
            self._skip_unit(u_idx)
            return

        # MANEUVER move
        if A_MOVE_START <= action < A_MOVE_END:
            idx = action - A_MOVE_START
            u_idx = idx // NUM_MOVE_OFFSETS
            off_idx = idx % NUM_MOVE_OFFSETS
            self._do_move(u_idx, off_idx)
            self._maybe_auto_advance()
            return

        # DEPLOY
        if A_DEPLOY_START <= action < A_DEPLOY_END:
            idx = action - A_DEPLOY_START
            u_idx = idx // NUM_DEPLOY_SLOTS
            slot_idx = idx % NUM_DEPLOY_SLOTS
            self._do_deploy(u_idx, slot_idx)
            return

        # RANGED
        if A_RANGED_START <= action < A_RANGED_END:
            idx = action - A_RANGED_START
            u_idx = idx // NUM_TARGETS
            t_idx = idx % NUM_TARGETS
            self._do_shoot(u_idx, t_idx)
            self._maybe_auto_advance()
            return

        # ENGAGE (charge)
        if A_ENGAGE_START <= action < A_ENGAGE_END:
            idx = action - A_ENGAGE_START
            u_idx = idx // NUM_TARGETS
            t_idx = idx % NUM_TARGETS
            self._do_charge(u_idx, t_idx)
            self._maybe_auto_advance()
            return

        # MELEE (fight)
        if A_MELEE_START <= action < A_MELEE_END:
            idx = action - A_MELEE_START
            u_idx = idx // NUM_TARGETS
            t_idx = idx % NUM_TARGETS
            self._do_fight(u_idx, t_idx)
            self._maybe_auto_advance()
            return

    # ─── Sub-action implementations ────────────────────────────────────────

    def _skip_unit(self, u_idx: int) -> None:
        """Mark a unit as having acted in the current phase (no effect)."""
        st = self.state
        unit = self._get_friendly(u_idx)
        if unit is None or unit.is_dead or not unit.deployed:
            return
        if st.phase == Phase.MANEUVER:
            unit.has_moved = True
        elif st.phase == Phase.RANGED:
            unit.has_shot = True
        elif st.phase == Phase.ENGAGE:
            unit.has_charged = True
        elif st.phase == Phase.MELEE:
            unit.has_fought = True

    def _do_move(self, u_idx: int, off_idx: int) -> None:
        st = self.state; bd = self.board
        unit = self._get_friendly(u_idx)
        if unit is None or unit.is_dead or not unit.deployed or unit.has_moved:
            return

        # Cannot move if engaged (would be a Fall Back; simplified: not used by agent here)
        if st.is_engaged(unit):
            unit.has_moved = True
            return

        # Compute target
        dx, dy = _MOVE_GRID[off_idx]
        mv = unit.movement + (unit.advance_bonus or 0)
        tx = unit.x + dx * mv
        ty = unit.y + dy * mv
        tx, ty = bd.clamp(tx, ty, r=unit.radius_in())
        tx, ty = bd.snap_to_grid(tx, ty)

        if not bd.can_move_to(unit.x, unit.y, tx, ty, mv, unit.is_infantry):
            unit.has_moved = True
            return
        if not st.is_position_clear(tx, ty, exclude=[unit], moving_unit=unit):
            unit.has_moved = True
            return

        unit.x = tx; unit.y = ty
        unit.has_moved = True

    def _do_deploy(self, u_idx: int, slot_idx: int) -> None:
        st = self.state
        bank = st.bank_units(st.active_team)
        if not bank:
            return
        # u_idx indexes into the team's roster; fall back to bank position
        unit = self._get_friendly(u_idx)
        if unit is None or unit.deployed or unit.is_dead:
            # Try interpreting u_idx as bank order
            if u_idx < len(bank):
                unit = bank[u_idx]
            else:
                return

        z = deploy_zone(st.active_team, st.deploy_style)
        fx, fy = _DEPLOY_GRID[slot_idx]
        x = z["x"] + fx * z["w"]
        y = z["y"] + fy * z["h"]
        x, y = self.board.clamp(x, y, r=unit.radius_in())
        x, y = self.board.snap_to_grid(x, y)

        ok = st.deploy_unit(unit, x, y)
        if not ok:
            # Search nearby for a legal slot
            for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                             (1, 1), (-1, -1), (-1, 1), (1, -1),
                             (2, 0), (-2, 0), (0, 2), (0, -2)]:
                nx = x + ddx * 1.5
                ny = y + ddy * 1.5
                nx, ny = self.board.clamp(nx, ny, r=unit.radius_in())
                nx, ny = self.board.snap_to_grid(nx, ny)
                if st.deploy_unit(unit, nx, ny):
                    return

    def _do_shoot(self, u_idx: int, t_idx: int) -> None:
        st = self.state; bd = self.board
        attacker = self._get_friendly(u_idx)
        if attacker is None or attacker.is_dead or attacker.has_shot:
            return
        targets = st.valid_shoot_targets(attacker, bd)
        target = self._get_enemy_in_team_slot(t_idx)
        if target is None or target not in targets:
            return
        dist = bd.dist_inches(attacker.x, attacker.y, target.x, target.y)
        before_hp = target.wounds_remaining
        res = attacker.shoot(target, dist, st.rng)
        res["type"] = "shoot"
        st.log({"type": "shoot", **res})

        dmg = before_hp - target.wounds_remaining
        self._record_damage(attacker.team, dmg, target_died=target.is_dead)

    def _do_charge(self, u_idx: int, t_idx: int) -> None:
        st = self.state; bd = self.board
        attacker = self._get_friendly(u_idx)
        if attacker is None or attacker.is_dead or attacker.has_charged:
            return
        targets = st.valid_charge_targets(attacker, bd)
        target = self._get_enemy_in_team_slot(t_idx)
        if target is None or target not in targets:
            return
        dist = bd.dist_inches(attacker.x, attacker.y, target.x, target.y)
        res = attacker.roll_charge(target, dist, st.rng)
        if res["success"]:
            pos = st.find_charge_contact_position(attacker, target, bd)
            if pos is not None:
                attacker.x, attacker.y = pos
                attacker.engaged_with = target
                target.engaged_with = attacker
            else:
                res["success"] = False
                res["blocked"] = True
        res["type"] = "charge"
        st.log({"type": "charge", **res})

    def _do_fight(self, u_idx: int, t_idx: int) -> None:
        st = self.state
        attacker = self._get_friendly(u_idx)
        if attacker is None or attacker.is_dead or attacker.has_fought:
            return
        targets = st.valid_fight_targets(attacker)
        target = self._get_enemy_in_team_slot(t_idx)
        if target is None or target not in targets:
            return
        before_hp = target.wounds_remaining
        res = attacker.fight(target, st.rng)
        res["type"] = "fight"
        st.log({"type": "fight", **res})
        dmg = before_hp - target.wounds_remaining
        self._record_damage(attacker.team, dmg, target_died=target.is_dead)

    # ─── Damage / VP accounting ────────────────────────────────────────────

    def _record_damage(self, attacker_team: int, dmg: int, target_died: bool) -> None:
        if dmg <= 0:
            return
        defender_team = 1 - attacker_team
        self._cum_dmg_dealt[attacker_team] += dmg
        self._cum_dmg_taken[defender_team] += dmg
        if target_died:
            self._cum_kills[attacker_team] += 1
            self._cum_losses[defender_team] += 1

    def _score_and_update_vp(self) -> None:
        """Score objectives, then sync _cum_vp from state."""
        if self.state.mode != GameMode.OBJECTIVES:
            return
        self._cum_vp = list(self.state.vp)

    # ─── Phase advancement ─────────────────────────────────────────────────

    def _end_phase(self) -> None:
        st = self.state; bd = self.board
        # If we're closing the MELEE phase, resolve counter-attacks first
        if st.phase == Phase.MELEE:
            counters = st.collect_counter_attacks()
            for c in counters:
                before_snapshot = {u.id: u.wounds_remaining for u in st.units
                                   if u.team == st.active_team}
                res = c()
                if res is None:
                    continue
                # Credit damage dealt by counter-team (1 - active_team) to itself
                counter_team = 1 - st.active_team
                after_hp = {u.id: u.wounds_remaining for u in st.units
                            if u.team == st.active_team}
                delta_dmg = 0
                died = 0
                for uid, before_hp in before_snapshot.items():
                    dmg = before_hp - after_hp.get(uid, before_hp)
                    if dmg > 0:
                        delta_dmg += dmg
                        # Was the unit alive before and dead now?
                        u_after = next((u for u in st.units if u.id == uid), None)
                        if u_after is not None and u_after.is_dead:
                            died += 1
                if delta_dmg > 0:
                    self._cum_dmg_dealt[counter_team] += delta_dmg
                    self._cum_dmg_taken[st.active_team] += delta_dmg
                if died > 0:
                    self._cum_kills[counter_team] += died
                    self._cum_losses[st.active_team] += died

        if st.phase == Phase.DEPLOY:
            # In deploy, end_phase shouldn't be picked — deployment auto-ends.
            # If invoked, force alternation to the other team without deploying.
            # (This is defensive only; mask normally disallows end_phase during deploy.)
            return

        st.next_phase(bd)
        # Update VP after round transitions (scoring happens inside next_phase)
        self._score_and_update_vp()

    def _maybe_auto_advance(self) -> None:
        """If the active team has no remaining legal actions for the current
        phase (all relevant units are flagged), silently advance phase."""
        st = self.state
        if st.phase == Phase.DEPLOY:
            return
        # Check if any useful action remains
        has_action = False
        for u in st.friendly_units():
            if st.phase == Phase.MANEUVER and not u.has_moved: has_action = True; break
            if st.phase == Phase.RANGED   and not u.has_shot and not st.is_engaged(u): has_action = True; break
            if st.phase == Phase.ENGAGE   and not u.has_charged and not st.is_engaged(u): has_action = True; break
            if st.phase == Phase.MELEE    and not u.has_fought and (u.has_charged or st.is_engaged(u)): has_action = True; break
        if not has_action:
            self._end_phase()

    # ─── Roster helpers ────────────────────────────────────────────────────

    def _get_friendly(self, u_idx: int) -> Unit | None:
        """Active-team unit at slot u_idx (0..MAX-1). Returns None if empty."""
        roster = self._team_units[self.state.active_team]
        if u_idx < 0 or u_idx >= len(roster):
            return None
        return roster[u_idx]

    def _get_enemy_in_team_slot(self, t_idx: int) -> Unit | None:
        other = 1 - self.state.active_team
        roster = self._team_units[other]
        if t_idx < 0 or t_idx >= len(roster):
            return None
        u = roster[t_idx]
        if u.is_dead:
            return None
        return u

    # ─── Observation ───────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        st = self.state
        me = st.active_team
        them = 1 - me
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # Globals
        off = 0
        phase_idx = {"DEPLOY": 0, "MANEUVER": 1, "RANGED": 2, "ENGAGE": 3, "MELEE": 4}[st.phase]
        obs[off + phase_idx] = 1.0
        off += NUM_PHASES

        obs[off] = st.turn / MAX_TURNS; off += 1
        obs[off] = 1.0; off += 1  # active-team flag (always me)
        obs[off] = min(1.0, st.vp[me] / 10.0); off += 1
        obs[off] = min(1.0, st.vp[them] / 10.0); off += 1
        obs[off] = 1.0 if st.mode == GameMode.OBJECTIVES else 0.0; off += 1

        my_f = self.team_factions[me]
        their_f = self.team_factions[them]
        obs[off + FACTION_KEYS.index(my_f)] = 1.0; off += NUM_FACTIONS
        obs[off + FACTION_KEYS.index(their_f)] = 1.0; off += NUM_FACTIONS

        style_idx = {"standard": 0, "frontal": 1, "diagonal": 2}[st.deploy_style]
        obs[off + style_idx] = 1.0; off += NUM_DEPLOY_STYLES

        # Objectives
        for i in range(NUM_OBJECTIVES):
            if i < len(st.objectives):
                o = st.objectives[i]
                obs[off] = o["x"] / BOARD_W; off += 1
                obs[off] = o["y"] / BOARD_H; off += 1
                ctrl = self.board.controlling_team(o["x"], o["y"], st.alive_units())
                if ctrl == me:   obs[off] = 1.0
                elif ctrl == them: obs[off + 1] = 1.0
                else:            obs[off + 2] = 1.0
                off += 3
            else:
                off += 5

        # My units
        for u in self._team_units[me]:
            self._fill_unit_features(obs, off, u, is_me=True)
            off += UNIT_FEATURES
        # Pad
        off += (MAX_UNITS_PER_TEAM - len(self._team_units[me])) * UNIT_FEATURES

        # Their units
        for u in self._team_units[them]:
            self._fill_unit_features(obs, off, u, is_me=False)
            off += UNIT_FEATURES
        off += (MAX_UNITS_PER_TEAM - len(self._team_units[them])) * UNIT_FEATURES

        assert off == OBS_DIM, f"obs offset mismatch {off} vs {OBS_DIM}"
        return obs

    def _fill_unit_features(self, obs: np.ndarray, off: int, u: Unit, is_me: bool) -> None:
        st = self.state
        i = 0
        obs[off + i] = 0.0 if u.is_dead else 1.0; i += 1
        obs[off + i] = u.hp_fraction(); i += 1
        obs[off + i] = u.x / BOARD_W; i += 1
        obs[off + i] = u.y / BOARD_H; i += 1
        obs[off + i] = u.movement / NORM_MOVEMENT; i += 1
        obs[off + i] = u.toughness / NORM_TOUGH; i += 1
        obs[off + i] = u.weapon_skill / NORM_WS; i += 1
        obs[off + i] = u.ballistic_skill / NORM_BS; i += 1
        obs[off + i] = u.wounds / NORM_WOUNDS; i += 1
        obs[off + i] = u.save / NORM_SAVE; i += 1
        obs[off + i] = (u.nullfield_save / NORM_SAVE) if u.nullfield_save else 0.0; i += 1
        obs[off + i] = u.objective_control / NORM_OBJ_CTRL; i += 1
        obs[off + i] = u.ranged.range / NORM_RANGE; i += 1
        obs[off + i] = u.ranged.attacks / NORM_ATTACKS; i += 1
        obs[off + i] = u.ranged.strength / NORM_STRENGTH; i += 1
        obs[off + i] = abs(u.ranged.ap) / NORM_AP; i += 1
        obs[off + i] = u.ranged.damage / NORM_DAMAGE; i += 1
        obs[off + i] = u.melee.attacks / NORM_ATTACKS; i += 1
        obs[off + i] = u.melee.strength / NORM_STRENGTH; i += 1
        obs[off + i] = abs(u.melee.ap) / NORM_AP; i += 1
        obs[off + i] = u.melee.damage / NORM_DAMAGE; i += 1
        obs[off + i] = 1.0 if u.has_moved else 0.0; i += 1
        obs[off + i] = 1.0 if u.has_shot else 0.0; i += 1
        obs[off + i] = 1.0 if u.has_charged else 0.0; i += 1
        obs[off + i] = 1.0 if u.has_fought else 0.0; i += 1
        obs[off + i] = 1.0 if u.has_fallen_back else 0.0; i += 1
        obs[off + i] = 1.0 if u.deployed else 0.0; i += 1
        obs[off + i] = 1.0 if u.is_infantry else 0.0; i += 1
        obs[off + i] = 1.0 if is_me else 0.0; i += 1
        obs[off + i] = 1.0 if st.is_engaged(u) else 0.0; i += 1
        assert i == UNIT_FEATURES, f"unit feature count {i} vs {UNIT_FEATURES}"

    # ─── Action mask ───────────────────────────────────────────────────────

    def _build_mask(self) -> np.ndarray:
        st = self.state; bd = self.board
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)

        friendly = self._team_units[st.active_team]

        if st.phase == Phase.DEPLOY:
            # Only deploy actions for units still in bank; end_phase disabled.
            for i, u in enumerate(friendly):
                if u.deployed or u.is_dead or i >= MAX_UNITS_PER_TEAM:
                    continue
                # Any slot is a candidate; deploy() will reject overlaps
                for s in range(NUM_DEPLOY_SLOTS):
                    a = A_DEPLOY_START + i * NUM_DEPLOY_SLOTS + s
                    mask[a] = 1
            return mask

        # Non-deploy phases: end_phase always available
        mask[A_END_PHASE] = 1

        for i, u in enumerate(friendly):
            if i >= MAX_UNITS_PER_TEAM or u.is_dead or not u.deployed:
                continue

            if st.phase == Phase.MANEUVER:
                if u.has_moved or st.is_engaged(u):
                    continue
                mask[A_SKIP_UNIT_START + i] = 1
                # Movement offsets: legality is checked at apply time; allow all
                for o in range(NUM_MOVE_OFFSETS):
                    mask[A_MOVE_START + i * NUM_MOVE_OFFSETS + o] = 1

            elif st.phase == Phase.RANGED:
                if u.has_shot or st.is_engaged(u):
                    continue
                if u.has_fallen_back and not u.can_fall_back_and_shoot:
                    continue
                targets = st.valid_shoot_targets(u, bd)
                if not targets:
                    continue
                mask[A_SKIP_UNIT_START + i] = 1
                enemy_roster = self._team_units[1 - st.active_team]
                for t in targets:
                    try:
                        t_idx = enemy_roster.index(t)
                    except ValueError:
                        continue
                    if t_idx >= NUM_TARGETS:
                        continue
                    mask[A_RANGED_START + i * NUM_TARGETS + t_idx] = 1

            elif st.phase == Phase.ENGAGE:
                if u.has_charged or u.has_fallen_back or st.is_engaged(u):
                    continue
                targets = st.valid_charge_targets(u, bd)
                if not targets:
                    continue
                mask[A_SKIP_UNIT_START + i] = 1
                enemy_roster = self._team_units[1 - st.active_team]
                for t in targets:
                    try:
                        t_idx = enemy_roster.index(t)
                    except ValueError:
                        continue
                    if t_idx >= NUM_TARGETS:
                        continue
                    mask[A_ENGAGE_START + i * NUM_TARGETS + t_idx] = 1

            elif st.phase == Phase.MELEE:
                if u.has_fought:
                    continue
                if not u.has_charged and not st.is_engaged(u):
                    continue
                targets = st.valid_fight_targets(u)
                if not targets:
                    continue
                mask[A_SKIP_UNIT_START + i] = 1
                enemy_roster = self._team_units[1 - st.active_team]
                for t in targets:
                    try:
                        t_idx = enemy_roster.index(t)
                    except ValueError:
                        continue
                    if t_idx >= NUM_TARGETS:
                        continue
                    mask[A_MELEE_START + i * NUM_TARGETS + t_idx] = 1

        return mask

    # ─── Reward bookkeeping ────────────────────────────────────────────────

    def _empty_snap(self) -> dict:
        return dict(dmg_dealt=0, dmg_taken=0, kills=0, losses=0, vp=0, terminal_applied=False)

    def _pop_reward(self, team: int, done: bool) -> float:
        """Compute reward since `team` last observed; update their snapshot."""
        snap = self._snap[team]
        cfg = self.reward_cfg
        r = 0.0
        r += cfg.dmg_dealt    * (self._cum_dmg_dealt[team] - snap["dmg_dealt"])
        r -= cfg.dmg_taken    * (self._cum_dmg_taken[team] - snap["dmg_taken"])
        r += cfg.enemy_killed * (self._cum_kills[team]     - snap["kills"])
        r -= cfg.own_killed   * (self._cum_losses[team]    - snap["losses"])

        cur_vp = self.state.vp[team] if self.state else 0
        enemy_vp = self.state.vp[1 - team] if self.state else 0
        prev_vp = snap["vp"]
        prev_enemy_vp = snap.get("enemy_vp", 0)
        r += cfg.vp_gained   * max(0, cur_vp - prev_vp)
        r -= cfg.vp_conceded * max(0, enemy_vp - prev_enemy_vp)

        if done and not snap["terminal_applied"]:
            if self.state.winner == team:
                r += cfg.win
            elif self.state.winner == 1 - team:
                r -= cfg.loss
            snap["terminal_applied"] = True

        snap["dmg_dealt"] = self._cum_dmg_dealt[team]
        snap["dmg_taken"] = self._cum_dmg_taken[team]
        snap["kills"]     = self._cum_kills[team]
        snap["losses"]    = self._cum_losses[team]
        snap["vp"]        = cur_vp
        snap["enemy_vp"]  = enemy_vp
        return float(r)

    # ─── Rendering ─────────────────────────────────────────────────────────

    def render(self, mode: str = "ascii") -> str:
        """ASCII renderer for debugging — one char per 2×2 inch cell."""
        st = self.state; bd = self.board
        if st is None or bd is None:
            return "[env not reset]"
        cell = 2.0
        cols = int(BOARD_W / cell)
        rows = int(BOARD_H / cell)
        grid = [["." for _ in range(cols)] for _ in range(rows)]

        # Terrain
        for t in bd.terrain:
            s = t.shape()
            for r in range(rows):
                for c in range(cols):
                    cx = (c + 0.5) * cell
                    cy = (r + 0.5) * cell
                    from .board import point_in_shape as _pis
                    if _pis(cx, cy, s):
                        grid[r][c] = "#" if t.blocks_move else "+"

        # Objectives
        for o in st.objectives:
            cc = int(o["x"] / cell); rr = int(o["y"] / cell)
            if 0 <= cc < cols and 0 <= rr < rows:
                grid[rr][cc] = o["name"]

        # Units
        for u in st.units:
            if u.is_dead or not u.deployed:
                continue
            cc = int(u.x / cell); rr = int(u.y / cell)
            if 0 <= cc < cols and 0 <= rr < rows:
                grid[rr][cc] = ("0" if u.team == 0 else "1")

        lines = [f"Turn {st.turn}  Phase {st.phase}  Team {st.active_team}  VP {st.vp}"]
        lines += ["".join(row) for row in grid]
        return "\n".join(lines)
