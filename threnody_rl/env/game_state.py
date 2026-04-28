"""GameState — turn structure, phase management, VP scoring.

Port of `src/core/GameState.js`. Pure logic, no rendering.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from .unit import Unit
from .board import Board


class Phase:
    DEPLOY   = "DEPLOY"
    MANEUVER = "MANEUVER"
    RANGED   = "RANGED"
    ENGAGE   = "ENGAGE"
    MELEE    = "MELEE"


class GameMode:
    DEATHMATCH = "DEATHMATCH"
    OBJECTIVES = "OBJECTIVES"


MAX_TURNS = 5
CONTACT_BUFFER = 0.15
BASE_CONTACT_RANGE = 1.3   # legacy; prefer contact_range(a, b)


def contact_range(a: Unit, b: Unit) -> float:
    return a.radius_in() + b.radius_in() + CONTACT_BUFFER


def deploy_zone(team: int, style: str) -> dict:
    """Deployment zone rectangle (inches) — matches JS deployZone().

    60×44 landscape board. STANDARD zones are 18×44 flank strips with a
    24" centre gap (bumped from 12" in Pass-3 tuning)."""
    if style == "diagonal":
        if team == 0:
            return {"x": 0, "y": 22, "w": 22, "h": 22}
        return {"x": 38, "y": 0, "w": 22, "h": 22}
    if style == "frontal":
        if team == 0:
            return {"x": 0, "y": 32, "w": 60, "h": 12}
        return {"x": 0, "y": 0, "w": 60, "h": 12}
    # STANDARD (FLANK)
    if team == 0:
        return {"x": 0, "y": 0, "w": 18, "h": 44}
    return {"x": 42, "y": 0, "w": 18, "h": 44}


# ─── GameState ───────────────────────────────────────────────────────────────

class GameState:
    def __init__(self, units: list[Unit],
                 objectives: list[dict] | None = None,
                 mode: str = GameMode.DEATHMATCH,
                 deploy_style: str = "standard",
                 rng: random.Random | None = None):
        self.units = units
        self.objectives = objectives or []
        self.mode = mode
        self.deploy_style = deploy_style
        self.rng = rng if rng is not None else random.Random()

        self.deploy_first_team = self.rng.randint(0, 1)
        self.first_team = self.deploy_first_team
        self.turn = 1
        self.active_team = self.deploy_first_team
        self.phase = Phase.DEPLOY
        self.vp = [0, 0]
        self.game_over = False
        self.winner: int | None = None

        for u in self.units:
            u.deployed = False

        self.combat_log: list[dict] = []

    # ─── Deployment ─────────────────────────────────────────────────────────

    def bank_units(self, team: int) -> list[Unit]:
        return [u for u in self.units if u.team == team and not u.deployed and not u.is_dead]

    def in_deployment_zone(self, team: int, x: float, y: float) -> bool:
        z = deploy_zone(team, self.deploy_style)
        return z["x"] <= x < z["x"] + z["w"] and z["y"] <= y < z["y"] + z["h"]

    def deploy_unit(self, unit: Unit, x: float, y: float) -> bool:
        if self.phase != Phase.DEPLOY: return False
        if unit.team != self.active_team: return False
        if unit.deployed: return False
        if not self.in_deployment_zone(unit.team, x, y): return False

        # No overlap with already-deployed units
        for other in self.units:
            if not other.deployed or other.is_dead:
                continue
            dx = other.x - x; dy = other.y - y
            if math.hypot(dx, dy) < other.radius_in() + unit.radius_in() + 0.1:
                return False

        unit.x = x; unit.y = y; unit.deployed = True

        # Alternate to other team if they still have units to place
        other_team = 1 - self.active_team
        if len(self.bank_units(other_team)) > 0:
            self.active_team = other_team
        elif len(self.bank_units(self.active_team)) == 0:
            self.end_deployment()
        return True

    def all_units_deployed(self) -> bool:
        return all(u.deployed or u.is_dead for u in self.units)

    def end_deployment(self) -> str:
        # Re-roll initiative so last-placer isn't automatically rewarded/penalised
        self.first_team = self.rng.randint(0, 1)
        self.active_team = self.first_team
        self.phase = Phase.MANEUVER
        self._start_team_turn()
        return f"team{self.first_team} wins initiative — Round 1 Maneuver Phase"

    # ─── Accessors ──────────────────────────────────────────────────────────

    def friendly_units(self) -> list[Unit]:
        return [u for u in self.units if u.team == self.active_team and not u.is_dead]

    def enemy_units(self) -> list[Unit]:
        return [u for u in self.units if u.team != self.active_team and not u.is_dead]

    def alive_units(self) -> list[Unit]:
        return [u for u in self.units if not u.is_dead]

    def team0_alive(self) -> list[Unit]: return [u for u in self.units if u.team == 0 and not u.is_dead]
    def team1_alive(self) -> list[Unit]: return [u for u in self.units if u.team == 1 and not u.is_dead]

    # ─── Phase advancement ──────────────────────────────────────────────────

    def next_phase(self, board: Board) -> str:
        if self.phase == Phase.MANEUVER:
            self.phase = Phase.RANGED
            return f"team{self.active_team}: Ranged Phase"
        if self.phase == Phase.RANGED:
            self.phase = Phase.ENGAGE
            return f"team{self.active_team}: Engage Phase"
        if self.phase == Phase.ENGAGE:
            self.phase = Phase.MELEE
            return f"team{self.active_team}: Melee Phase"

        # MELEE → end team's turn
        self._end_team_turn(board)
        if self.game_over:
            return f"Game over — winner {self.winner}"

        if self.active_team == self.first_team:
            self.active_team = 1 - self.first_team
            self.phase = Phase.MANEUVER
            self._start_team_turn()
            return f"team{self.active_team}: Maneuver Phase"
        else:
            self.turn += 1
            if self.turn > MAX_TURNS:
                self._resolve_game_end()
                return f"Game over — winner {self.winner}"
            self.active_team = self.first_team
            self.phase = Phase.MANEUVER
            self._score_objectives(board)
            self._start_team_turn()
            return f"Battle Round {self.turn} — team{self.active_team}: Maneuver Phase"

    def _start_team_turn(self) -> None:
        for u in self.units:
            if u.team == self.active_team:
                u.reset_turn()

    def _end_team_turn(self, board: Board) -> None:
        if len(self.team0_alive()) == 0:
            self.game_over = True; self.winner = 1; return
        if len(self.team1_alive()) == 0:
            self.game_over = True; self.winner = 0; return

    def _score_objectives(self, board: Board) -> None:
        if self.mode != GameMode.OBJECTIVES:
            return
        for obj in self.objectives:
            ctrl = board.controlling_team(obj["x"], obj["y"], self.alive_units())
            if ctrl >= 0:
                self.vp[ctrl] += 1

    def _resolve_game_end(self) -> None:
        self.game_over = True
        if self.mode == GameMode.DEATHMATCH:
            a = len(self.team0_alive()); b = len(self.team1_alive())
            self.winner = 0 if a > b else 1 if b > a else -1
        else:
            if self.vp[0] > self.vp[1]:
                self.winner = 0
            elif self.vp[1] > self.vp[0]:
                self.winner = 1
            else:
                self.winner = -1

    # ─── Unit queries used by AI ────────────────────────────────────────────

    def _center_dist(self, a: Unit, b: Unit) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def in_base_contact(self, a: Unit, b: Unit) -> bool:
        return self._center_dist(a, b) <= contact_range(a, b)

    def is_fall_back_position(self, x: float, y: float, unit: Unit) -> bool:
        my_r = unit.radius_in()
        for u in self.units:
            if u.is_dead or u.team == unit.team:
                continue
            if math.hypot(u.x - x, u.y - y) <= my_r + u.radius_in() + CONTACT_BUFFER:
                return False
        return True

    def is_position_clear(self, x: float, y: float,
                          exclude: list[Unit] | None = None,
                          moving_unit: Unit | None = None) -> bool:
        exclude = exclude or []
        my_r = moving_unit.radius_in() if moving_unit else 0.6
        for u in self.units:
            if u.is_dead:
                continue
            if u in exclude:
                continue
            if math.hypot(u.x - x, u.y - y) < my_r + u.radius_in():
                return False
        return True

    def find_charge_contact_position(self, attacker: Unit, target: Unit,
                                     board: Board) -> tuple[float, float] | None:
        touch_dist = attacker.radius_in() + target.radius_in() + 0.05
        NUM_ANGLES = 48
        base_angle = math.atan2(attacker.y - target.y, attacker.x - target.x)
        step = (math.pi * 2) / NUM_ANGLES

        for i in range(NUM_ANGLES):
            offset = ((i + 1) // 2) * step   # 0, step, step, 2*step, 2*step, ...
            sign = 1 if i % 2 == 0 else -1
            angle = base_angle + sign * offset
            x = target.x + math.cos(angle) * touch_dist
            y = target.y + math.sin(angle) * touch_dist
            if x < 0 or x > board.width_in or y < 0 or y > board.height_in:
                continue
            if not self.is_position_clear(x, y, [attacker, target], attacker):
                continue
            return (x, y)
        return None

    def is_engaged(self, unit: Unit) -> bool:
        return any(
            (not u.is_dead and u.team != unit.team and self.in_base_contact(u, unit))
            for u in self.units
        )

    def can_fight(self, attacker: Unit, target: Unit) -> bool:
        if self.in_base_contact(attacker, target):
            return True
        for ally in self.units:
            if ally.is_dead or ally is attacker:
                continue
            if ally.team != attacker.team:
                continue
            if self.in_base_contact(attacker, ally) and self.in_base_contact(ally, target):
                return True
        return False

    def valid_shoot_targets(self, unit: Unit, board: Board) -> list[Unit]:
        if self.phase != Phase.RANGED: return []
        if unit.has_shot: return []
        if self.is_engaged(unit): return []
        if unit.has_fallen_back and not unit.can_fall_back_and_shoot: return []

        out = []
        for t in self.units:
            if t.is_dead or t.team == unit.team:
                continue
            d = board.dist_inches(unit.x, unit.y, t.x, t.y)
            if d > unit.ranged.range:
                continue
            if not board.has_los(unit.x, unit.y, t.x, t.y):
                continue
            out.append(t)
        return out

    def valid_charge_targets(self, unit: Unit, board: Board) -> list[Unit]:
        if self.phase != Phase.ENGAGE: return []
        if unit.has_charged: return []
        if unit.has_fallen_back: return []
        if self.is_engaged(unit): return []

        return [
            t for t in self.units
            if (not t.is_dead and t.team != unit.team and
                board.dist_inches(unit.x, unit.y, t.x, t.y) <= 12)
        ]

    def valid_fight_targets(self, unit: Unit) -> list[Unit]:
        if self.phase != Phase.MELEE: return []
        if unit.has_fought: return []
        if not unit.has_charged and not self.is_engaged(unit): return []
        return [
            t for t in self.units
            if (not t.is_dead and t.team != unit.team and self.can_fight(unit, t))
        ]

    def collect_counter_attacks(self) -> list[Any]:
        """Return callables; the env iterates them after MELEE phase closes."""
        if self.phase != Phase.MELEE:
            return []
        counter_side = 1 - self.active_team
        actions = []
        for u in self.units:
            if u.team != counter_side or u.is_dead:
                continue
            targets = [t for t in self.units
                       if t.team == self.active_team and not t.is_dead
                       and self.in_base_contact(u, t)]
            if not targets:
                continue

            def make_action(attacker: Unit):
                def run():
                    if attacker.is_dead:
                        return None
                    live = [t for t in self.units
                            if t.team == self.active_team and not t.is_dead
                            and self.in_base_contact(attacker, t)]
                    if not live:
                        return None
                    tgt = min(live, key=lambda t: self._center_dist(attacker, t))
                    result = attacker.fight(tgt, self.rng)
                    result["type"] = "fight"; result["counter"] = True
                    self.log({"type": "fight", "counter": True, **result})
                    return result
                return run

            actions.append(make_action(u))
        return actions

    def log(self, entry: dict) -> None:
        self.combat_log.insert(0, entry)
        if len(self.combat_log) > 50:
            self.combat_log.pop()


# ─── Default objective placements (Stranglehold mirror) ──────────────────────

def default_objectives() -> list[dict]:
    """Three objectives placed to match GameScene.js Stranglehold layout.

    Triangulated off-centre so neither team has them all in their backfield.
    Coordinates synced 2026-04-28 to match GameScene.js:510-512 — B and C
    were drifting from a stale CLAUDE.md note (B was 18,11; C was 42,33)
    which inflated team-1's near-objective control. The current JS values
    spread B + C wider so neither team has both in their deploy zone."""
    return [
        {"x": 30, "y": 22, "name": "A"},
        {"x": 11, "y":  7, "name": "B"},
        {"x": 48, "y": 37, "name": "C"},
    ]
