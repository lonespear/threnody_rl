"""Unit — pure logic combat resolution.

Direct port of `src/core/Unit.js`. All positions in INCHES.
The dice / AP / nullfield-save semantics match the JS game exactly so
that statistical parity tests can verify the port.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


def d6(rng: random.Random | None = None) -> int:
    r = rng if rng is not None else random
    return r.randint(1, 6)


def roll_dice(n: int, rng: random.Random | None = None) -> list[int]:
    r = rng if rng is not None else random
    return [r.randint(1, 6) for _ in range(n)]


def wound_target(strength: int, toughness: int) -> int:
    if strength >= 2 * toughness:  return 2
    if strength > toughness:       return 3
    if strength == toughness:      return 4
    if toughness >= 2 * strength:  return 6
    return 5


@dataclass
class Weapon:
    name: str
    attacks: int
    strength: int
    ap: int
    damage: int
    range: int = 0
    keywords: list[str] = field(default_factory=list)


@dataclass
class Unit:
    # Identity
    id: str
    name: str
    faction: str
    team: int

    # Position (inches)
    x: float
    y: float

    # Core stats
    movement: float
    toughness: int
    weapon_skill: int
    ballistic_skill: int
    wounds: int
    save: int
    nullfield_save: int | None = None
    objective_control: int = 1

    ranged: Weapon = field(default_factory=lambda: Weapon("", 0, 0, 0, 0, 0, []))
    melee:  Weapon = field(default_factory=lambda: Weapon("", 0, 0, 0, 0, 0, []))

    is_infantry: bool = True
    base_size: float = 1.0

    # Abilities
    can_fall_back_and_shoot: bool = False
    advance_bonus: int = 0

    # Per-turn flags
    has_moved: bool = False
    has_shot: bool = False
    has_charged: bool = False
    has_fought: bool = False
    has_fallen_back: bool = False

    # State
    is_dead: bool = False
    deployed: bool = False
    wounds_remaining: int = 0
    engaged_with: Any = None

    def __post_init__(self):
        if self.wounds_remaining == 0:
            self.wounds_remaining = self.wounds

    # ─── Combat ────────────────────────────────────────────────────────────

    def shoot(self, target: "Unit", distance_to: float,
              rng: random.Random | None = None) -> dict[str, Any]:
        """Resolve ranged attack vs target. Mutates target HP + self.has_shot."""
        r = rng if rng is not None else random
        n_attacks = self.ranged.attacks

        # Volley Fire: extra attacks within half range
        rf = next((k for k in self.ranged.keywords if k.startswith("Volley Fire")), None)
        if rf and distance_to <= self.ranged.range / 2:
            parts = rf.split(" ")
            bonus = int(parts[2]) if len(parts) >= 3 else 1
            n_attacks += bonus

        res: dict[str, Any] = {
            "attacker": self.name, "defender": target.name,
            "weapon": self.ranged.name,
            "hits": 0, "wounds": 0, "unsaved": 0, "damage": 0,
        }

        hit_rolls = roll_dice(n_attacks, r)
        res["hits"] = sum(1 for x in hit_rolls if x >= self.ballistic_skill)
        if res["hits"] == 0:
            self.has_shot = True
            return res

        wt = wound_target(self.ranged.strength, target.toughness)
        wound_rolls = roll_dice(res["hits"], r)
        res["wounds"] = sum(1 for x in wound_rolls if x >= wt)
        if res["wounds"] == 0:
            self.has_shot = True
            return res

        mod_save = min(6, max(2, target.save - self.ranged.ap))
        best_save = (
            min(mod_save, target.nullfield_save)
            if target.nullfield_save is not None
            else mod_save
        )
        save_rolls = roll_dice(res["wounds"], r)
        res["unsaved"] = sum(1 for x in save_rolls if x < best_save)
        res["damage"] = res["unsaved"] * self.ranged.damage
        target.take_damage(res["damage"])
        self.has_shot = True
        return res

    def roll_charge(self, target: "Unit", distance_to: float,
                    rng: random.Random | None = None) -> dict[str, Any]:
        r = rng if rng is not None else random
        dice = [r.randint(1, 6), r.randint(1, 6)]
        total = sum(dice)
        self.has_charged = True
        return {
            "attacker": self.name, "defender": target.name,
            "distance": round(distance_to, 1),
            "roll": dice, "total": total,
            "success": total >= distance_to,
        }

    def fight(self, target: "Unit",
              rng: random.Random | None = None) -> dict[str, Any]:
        r = rng if rng is not None else random
        res: dict[str, Any] = {
            "attacker": self.name, "defender": target.name,
            "weapon": self.melee.name,
            "hits": 0, "wounds": 0, "unsaved": 0, "damage": 0,
        }

        hit_rolls = roll_dice(self.melee.attacks, r)
        res["hits"] = sum(1 for x in hit_rolls if x >= self.weapon_skill)
        if res["hits"] == 0:
            self.has_fought = True
            return res

        wt = wound_target(self.melee.strength, target.toughness)
        wound_rolls = roll_dice(res["hits"], r)
        res["wounds"] = sum(1 for x in wound_rolls if x >= wt)
        if res["wounds"] == 0:
            self.has_fought = True
            return res

        mod_save = min(6, max(2, target.save - self.melee.ap))
        best_save = (
            min(mod_save, target.nullfield_save)
            if target.nullfield_save is not None
            else mod_save
        )
        save_rolls = roll_dice(res["wounds"], r)
        res["unsaved"] = sum(1 for x in save_rolls if x < best_save)
        res["damage"] = res["unsaved"] * self.melee.damage
        target.take_damage(res["damage"])
        self.has_fought = True
        return res

    # ─── State ─────────────────────────────────────────────────────────────

    def take_damage(self, dmg: int) -> None:
        self.wounds_remaining = max(0, self.wounds_remaining - dmg)
        if self.wounds_remaining == 0:
            self.is_dead = True

    def reset_turn(self) -> None:
        self.has_moved = False
        self.has_shot = False
        self.has_charged = False
        self.has_fought = False
        self.has_fallen_back = False

    def hp_fraction(self) -> float:
        return self.wounds_remaining / self.wounds

    def radius_in(self) -> float:
        return 0.6 * 1.4 * self.base_size


# ─── Closed-form expected damage (for parity tests / AI scoring) ─────────────

def expected_damage_ranged(attacker: Unit, target: Unit, distance: float) -> float:
    """Match _killScore math in Opponent.js."""
    attacks = attacker.ranged.attacks
    rf = next((k for k in attacker.ranged.keywords if k.startswith("Volley Fire")), None)
    if rf and distance <= attacker.ranged.range / 2:
        parts = rf.split(" ")
        attacks += int(parts[2]) if len(parts) >= 3 else 1

    hit_prob = (7 - attacker.ballistic_skill) / 6
    wt = wound_target(attacker.ranged.strength, target.toughness)
    wound_prob = (7 - wt) / 6
    mod_save = min(6, max(2, target.save - attacker.ranged.ap))
    best_save = (min(mod_save, target.nullfield_save)
                 if target.nullfield_save is not None else mod_save)
    fail_save = (best_save - 1) / 6
    return attacks * hit_prob * wound_prob * fail_save * attacker.ranged.damage


def expected_damage_melee(attacker: Unit, target: Unit) -> float:
    attacks = attacker.melee.attacks
    hit_prob = (7 - attacker.weapon_skill) / 6
    wt = wound_target(attacker.melee.strength, target.toughness)
    wound_prob = (7 - wt) / 6
    mod_save = min(6, max(2, target.save - attacker.melee.ap))
    best_save = (min(mod_save, target.nullfield_save)
                 if target.nullfield_save is not None else mod_save)
    fail_save = (best_save - 1) / 6
    return attacks * hit_prob * wound_prob * fail_save * attacker.melee.damage
