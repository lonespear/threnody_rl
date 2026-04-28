"""Faction rosters + deployment styles — Python ports of src/data/*.js.

Stats match the canonical Pass-2 values (Render/Breaker/Phantom for Harrow,
Umbra/Skulker/Bloom for Drift, Drudge/Siphon/Magister for Voidborn, etc).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .unit import Unit, Weapon


BOARD_W = 60.0   # landscape long-edge (matches GameScene.js)
BOARD_H = 44.0


class DeployStyle:
    STANDARD = "standard"   # FLANK — left vs right
    FRONTAL  = "frontal"    # top vs bottom
    DIAGONAL = "diagonal"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _mirror_x(x: float) -> float: return BOARD_W - x
def _mirror_y(y: float) -> float: return BOARD_H - y


def _transform_deploy(base: list[dict], team: int, style: str) -> list[dict]:
    """Apply a deployment style to a base roster authored at small-x, small-y."""
    out = []
    for u in base:
        c = dict(u)
        c["team"] = team
        if style == DeployStyle.DIAGONAL:
            if team == 0:
                c["y"] = _mirror_y(c["y"])
            else:
                c["x"] = _mirror_x(c["x"])
        elif style == DeployStyle.FRONTAL:
            if team == 0:
                c["y"] = _mirror_y(c["y"])
        else:  # STANDARD (FLANK)
            if team == 1:
                c["x"] = _mirror_x(c["x"])
        out.append(c)
    return out


def _make_unit(cfg: dict) -> Unit:
    """Build a Unit from a factory cfg dict."""
    ranged_cfg = cfg["ranged"]
    melee_cfg  = cfg["melee"]
    return Unit(
        id=cfg["id"], name=cfg["name"], faction=cfg["faction"],
        team=cfg["team"],
        x=cfg["x"], y=cfg["y"],
        movement=cfg["movement"],
        toughness=cfg["toughness"],
        weapon_skill=cfg["weapon_skill"],
        ballistic_skill=cfg["ballistic_skill"],
        wounds=cfg["wounds"],
        save=cfg["save"],
        nullfield_save=cfg.get("nullfield_save"),
        objective_control=cfg.get("objective_control", 1),
        is_infantry=cfg.get("is_infantry", True),
        base_size=cfg.get("base_size", 1.0),
        can_fall_back_and_shoot=cfg.get("can_fall_back_and_shoot", False),
        advance_bonus=cfg.get("advance_bonus", 0),
        ranged=Weapon(
            name=ranged_cfg["name"],
            range=ranged_cfg["range"],
            attacks=ranged_cfg["attacks"],
            strength=ranged_cfg["strength"],
            ap=ranged_cfg["ap"],
            damage=ranged_cfg["damage"],
            keywords=list(ranged_cfg.get("keywords", [])),
        ),
        melee=Weapon(
            name=melee_cfg["name"],
            attacks=melee_cfg["attacks"],
            strength=melee_cfg["strength"],
            ap=melee_cfg["ap"],
            damage=melee_cfg["damage"],
        ),
    )


# ─── Accord Regulars ─────────────────────────────────────────────────────────

def _accord_line_trooper(uid, x, y):
    return dict(id=uid, name="Line Trooper", faction="accord", x=x, y=y,
                base_size=1.0, movement=6, toughness=3, weapon_skill=4,
                ballistic_skill=3, wounds=1, save=4, nullfield_save=None,
                objective_control=1, is_infantry=True,
                ranged=dict(name="Pulse Carbine", range=18, attacks=1,
                            strength=5, ap=-1, damage=1,
                            keywords=["Volley Fire 1"]),
                melee=dict(name="Combat Blade", attacks=1, strength=3,
                           ap=0, damage=1))


def _accord_shield_warden(uid, x, y):
    return dict(id=uid, name="Shield Warden", faction="accord", x=x, y=y,
                base_size=1.25, movement=5, toughness=4, weapon_skill=4,
                ballistic_skill=4, wounds=2, save=2, nullfield_save=4,
                objective_control=2, is_infantry=True,
                ranged=dict(name="Pulse Pistol", range=12, attacks=1,
                            strength=4, ap=-1, damage=1, keywords=[]),
                melee=dict(name="Field Shield Bash", attacks=2, strength=4,
                           ap=-1, damage=1))


def _accord_recon_strider(uid, x, y):
    return dict(id=uid, name="Recon Strider", faction="accord", x=x, y=y,
                base_size=1.25, movement=10, toughness=5, weapon_skill=4,
                ballistic_skill=3, wounds=4, save=3, nullfield_save=5,
                objective_control=1, is_infantry=False,
                can_fall_back_and_shoot=True,
                ranged=dict(name="Coil Array", range=24, attacks=2,
                            strength=7, ap=-2, damage=2, keywords=[]),
                melee=dict(name="Strider Fist", attacks=2, strength=5,
                           ap=-1, damage=1))


# ─── The Harrow ──────────────────────────────────────────────────────────────

def _harrow_render(uid, x, y):
    return dict(id=uid, name="Render", faction="harrow", x=x, y=y,
                base_size=1.0, movement=7, toughness=4, weapon_skill=3,
                ballistic_skill=5, wounds=2, save=4, nullfield_save=None,
                objective_control=2, is_infantry=True,
                ranged=dict(name="Frag Pistol", range=8, attacks=1,
                            strength=3, ap=0, damage=1, keywords=[]),
                melee=dict(name="Shear Claws", attacks=3, strength=5,
                           ap=-1, damage=1))


def _harrow_breaker(uid, x, y):
    return dict(id=uid, name="Breaker", faction="harrow", x=x, y=y,
                base_size=1.6, movement=5, toughness=6, weapon_skill=3,
                ballistic_skill=5, wounds=5, save=3, nullfield_save=None,
                objective_control=3, is_infantry=True, advance_bonus=3,
                ranged=dict(name="Thrown Debris", range=6, attacks=1,
                            strength=5, ap=0, damage=1, keywords=[]),
                melee=dict(name="Crushing Maul", attacks=4, strength=8,
                           ap=-2, damage=2))


def _harrow_phantom(uid, x, y):
    return dict(id=uid, name="Phantom", faction="harrow", x=x, y=y,
                base_size=1.0, movement=9, toughness=3, weapon_skill=2,
                ballistic_skill=5, wounds=2, save=5, nullfield_save=4,
                objective_control=1, is_infantry=True,
                ranged=dict(name="Spine Volley", range=10, attacks=2,
                            strength=3, ap=0, damage=1, keywords=[]),
                melee=dict(name="Blade Limbs", attacks=4, strength=4,
                           ap=-2, damage=1))


# ─── Ironveil Mandate ────────────────────────────────────────────────────────

def _ironveil_bulwark(uid, x, y):
    return dict(id=uid, name="Bulwark", faction="ironveil", x=x, y=y,
                base_size=1.25, movement=5, toughness=5, weapon_skill=4,
                ballistic_skill=3, wounds=3, save=2, nullfield_save=None,
                objective_control=2, is_infantry=True,
                ranged=dict(name="Arc Rifle", range=24, attacks=2,
                            strength=6, ap=-2, damage=2, keywords=[]),
                melee=dict(name="Powered Gauntlet", attacks=2, strength=5,
                           ap=-1, damage=1))


def _ironveil_siege_platform(uid, x, y):
    return dict(id=uid, name="Siege Platform", faction="ironveil", x=x, y=y,
                base_size=1.6, movement=4, toughness=6, weapon_skill=5,
                ballistic_skill=2, wounds=5, save=3, nullfield_save=None,
                objective_control=1, is_infantry=False,
                ranged=dict(name="Siege Cannon", range=36, attacks=3,
                            strength=8, ap=-3, damage=3, keywords=[]),
                melee=dict(name="Crushing Treads", attacks=1, strength=6,
                           ap=0, damage=1))


def _ironveil_aegis_warden(uid, x, y):
    return dict(id=uid, name="Aegis Warden", faction="ironveil", x=x, y=y,
                base_size=1.6, movement=5, toughness=7, weapon_skill=3,
                ballistic_skill=5, wounds=6, save=2, nullfield_save=4,
                objective_control=3, is_infantry=True,
                ranged=dict(name="Wrist Autogun", range=12, attacks=2,
                            strength=4, ap=0, damage=1, keywords=[]),
                melee=dict(name="Storm Halberd", attacks=4, strength=7,
                           ap=-2, damage=2))


# ─── The Drift ───────────────────────────────────────────────────────────────

def _drift_umbra(uid, x, y):
    return dict(id=uid, name="Umbra", faction="drift", x=x, y=y,
                base_size=1.0, movement=8, toughness=3, weapon_skill=5,
                ballistic_skill=4, wounds=1, save=6, nullfield_save=None,
                objective_control=1, is_infantry=True,
                ranged=dict(name="Shard Spray", range=12, attacks=2,
                            strength=3, ap=0, damage=1,
                            keywords=["Volley Fire 1"]),
                melee=dict(name="Thorn Hooks", attacks=2, strength=3,
                           ap=-1, damage=1))


def _drift_skulker(uid, x, y):
    return dict(id=uid, name="Skulker", faction="drift", x=x, y=y,
                base_size=1.25, movement=10, toughness=3, weapon_skill=2,
                ballistic_skill=5, wounds=2, save=5, nullfield_save=5,
                objective_control=1, is_infantry=True,
                ranged=dict(name="Venom Spit", range=8, attacks=1,
                            strength=4, ap=-1, damage=1, keywords=[]),
                melee=dict(name="Reaper Talons", attacks=5, strength=5,
                           ap=-2, damage=1))


def _drift_bloom(uid, x, y):
    return dict(id=uid, name="Bloom", faction="drift", x=x, y=y,
                base_size=1.25, movement=7, toughness=4, weapon_skill=5,
                ballistic_skill=3, wounds=3, save=5, nullfield_save=None,
                objective_control=2, is_infantry=True,
                can_fall_back_and_shoot=True,
                ranged=dict(name="Spore Burst", range=18, attacks=4,
                            strength=5, ap=-1, damage=1,
                            keywords=["Volley Fire 2"]),
                melee=dict(name="Lashing Tendrils", attacks=2, strength=4,
                           ap=0, damage=1))


# ─── Voidborn Cabal ──────────────────────────────────────────────────────────

def _voidborn_drudge(uid, x, y):
    return dict(id=uid, name="Drudge", faction="voidborn", x=x, y=y,
                base_size=1.0, movement=6, toughness=3, weapon_skill=4,
                ballistic_skill=3, wounds=2, save=5, nullfield_save=4,
                objective_control=1, is_infantry=True,
                ranged=dict(name="Cipher Bolt", range=18, attacks=2,
                            strength=6, ap=-2, damage=2, keywords=[]),
                melee=dict(name="Void Touch", attacks=2, strength=4,
                           ap=-1, damage=1))


def _voidborn_siphon(uid, x, y):
    return dict(id=uid, name="Siphon", faction="voidborn", x=x, y=y,
                base_size=1.25, movement=6, toughness=4, weapon_skill=5,
                ballistic_skill=2, wounds=4, save=4, nullfield_save=4,
                objective_control=2, is_infantry=True,
                ranged=dict(name="Drain Lance", range=24, attacks=4,
                            strength=7, ap=-3, damage=2, keywords=[]),
                melee=dict(name="Withering Grasp", attacks=1, strength=4,
                           ap=0, damage=1))


def _voidborn_magister(uid, x, y):
    return dict(id=uid, name="Magister", faction="voidborn", x=x, y=y,
                base_size=1.6, movement=7, toughness=4, weapon_skill=2,
                ballistic_skill=3, wounds=6, save=4, nullfield_save=3,
                objective_control=2, is_infantry=True,
                ranged=dict(name="Void Gaze", range=14, attacks=3,
                            strength=8, ap=-3, damage=3, keywords=[]),
                melee=dict(name="Phase Blade", attacks=5, strength=6,
                           ap=-3, damage=2))


# ─── Faction dispatch ────────────────────────────────────────────────────────

FACTION_KEYS = ("accord", "harrow", "ironveil", "drift", "voidborn")

FACTIONS: dict[str, dict] = {
    "accord": {
        "key": "accord", "name": "Accord Regulars",
        "tagline": "Balanced ranged military",
        "roster": lambda: [
            _accord_line_trooper("f0", 3, 5),
            _accord_line_trooper("f1", 3, 10),
            _accord_line_trooper("f2", 3, 15),
            _accord_line_trooper("f3", 3, 20),
            _accord_shield_warden("f4", 5, 25),
            _accord_recon_strider("f5", 5, 12),
        ],
    },
    "harrow": {
        "key": "harrow", "name": "The Harrow",
        "tagline": "Melee shock troops",
        "roster": lambda: [
            _harrow_render("f0", 3, 5),
            _harrow_render("f1", 3, 10),
            _harrow_render("f2", 3, 15),
            _harrow_render("f3", 3, 20),
            _harrow_breaker("f4", 5, 25),
            _harrow_phantom("f5", 5, 2),
        ],
    },
    "ironveil": {
        "key": "ironveil", "name": "Ironveil Mandate",
        "tagline": "Heavy armor / artillery",
        "roster": lambda: [
            _ironveil_bulwark("f0", 3, 6),
            _ironveil_bulwark("f1", 3, 14),
            _ironveil_bulwark("f2", 3, 22),
            _ironveil_siege_platform("f3", 5, 10),
            _ironveil_aegis_warden("f4", 4, 18),
        ],
    },
    "drift": {
        "key": "drift", "name": "The Drift",
        "tagline": "Fast glass-cannon swarm",
        "roster": lambda: [
            _drift_umbra("f0", 3, 3),
            _drift_umbra("f1", 3, 7),
            _drift_umbra("f2", 3, 11),
            _drift_umbra("f3", 3, 15),
            _drift_umbra("f4", 3, 19),
            _drift_skulker("f5", 4, 23),
            _drift_skulker("f6", 4, 27),
            _drift_bloom("f7", 5, 13),
        ],
    },
    "voidborn": {
        "key": "voidborn", "name": "Voidborn Cabal",
        "tagline": "Cipher elites / null-shielded",
        "roster": lambda: [
            _voidborn_drudge("f0", 3, 6),
            _voidborn_drudge("f1", 3, 14),
            _voidborn_drudge("f2", 3, 22),
            _voidborn_siphon("f3", 5, 10),
            _voidborn_magister("f4", 4, 18),
        ],
    },
}


def roster_for(faction_key: str, team: int, style: str = DeployStyle.STANDARD) -> list[Unit]:
    """Build a fresh list of Unit instances for a faction on a given team.

    Raises KeyError if `faction_key` is unknown. Team-1 unit IDs are
    prefixed so cross-team IDs don't collide."""
    if faction_key not in FACTIONS:
        raise KeyError(f"Unknown faction: {faction_key!r}")
    base_cfgs = FACTIONS[faction_key]["roster"]()
    transformed = _transform_deploy(base_cfgs, team, style)
    units: list[Unit] = []
    for cfg in transformed:
        cfg = dict(cfg)
        cfg["id"] = f"t{team}_{cfg['id']}"
        units.append(_make_unit(cfg))
    return units


def all_matchups() -> list[tuple[str, str]]:
    """Every (team0, team1) faction pairing."""
    return [(a, b) for a in FACTION_KEYS for b in FACTION_KEYS]


def max_units_per_team() -> int:
    """Largest roster size across factions (drift=8). Used to size obs/action tensors."""
    return max(len(FACTIONS[k]["roster"]()) for k in FACTION_KEYS)
