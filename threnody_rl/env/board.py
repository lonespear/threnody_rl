"""Board — world geometry, LOS, and movement pathfinding.

Direct Python port of `src/core/Board.js`. All positions in INCHES.
Terrain pieces can be axis-aligned rectangles (AABB), oriented bounding
boxes (OBB) with an `angle` field, or circles with a `r` field.

LOS: segment–shape intersection, with the convention that the piece
containing either endpoint is treated as transparent (a unit inside a
ruin can shoot out; a unit inside can be shot at).

Movement: Dijkstra flood on a fine grid (NAV_RES inches per cell).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


NAV_RES = 0.5          # inches per nav-grid cell
DIFFICULT_COST = 1.3   # infantry multiplier in difficult terrain; non-infantry pay even more


# ─── Terrain piece representation ────────────────────────────────────────────

@dataclass
class Terrain:
    """Axis-aligned rectangle (x,y,w,h) by default. For OBB set angle (deg).
    For circle set cx/cy/r and leave x/y/w/h as zero.
    A pre-baked `bbox` dict overrides (x,y,w,h) for LOS/move geometry."""
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    blocks_los: bool = True
    blocks_move: bool = True
    difficult: bool = False
    label: str = ""
    # OBB extras
    cx: float | None = None
    cy: float | None = None
    angle: float | None = None   # degrees; presence signals OBB
    # Circle extras
    r: float | None = None       # presence signals circle
    # Manual override shape (dict matching one of: aabb / obb / circle)
    bbox: dict[str, Any] | None = None

    def shape(self) -> dict[str, Any]:
        """Return the effective shape used for LOS + movement."""
        if self.bbox is not None:
            return self.bbox
        if self.r is not None:
            return {"cx": self.cx, "cy": self.cy, "r": self.r}
        if self.angle is not None:
            return {"cx": self.cx, "cy": self.cy, "w": self.w, "h": self.h, "angle": self.angle}
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


# ─── Shape geometry primitives ───────────────────────────────────────────────

def _point_in_rect(px: float, py: float, s: dict) -> bool:
    return s["x"] <= px <= s["x"] + s["w"] and s["y"] <= py <= s["y"] + s["h"]


def _point_in_circle(px: float, py: float, s: dict) -> bool:
    dx = px - s["cx"]; dy = py - s["cy"]
    return dx * dx + dy * dy <= s["r"] * s["r"]


def _to_obb_local(px: float, py: float, s: dict) -> tuple[float, float]:
    rad = math.radians(s["angle"])
    cosA = math.cos(rad); sinA = math.sin(rad)
    dx = px - s["cx"]; dy = py - s["cy"]
    return (dx * cosA + dy * sinA, -dx * sinA + dy * cosA)


def _point_in_obb(px: float, py: float, s: dict) -> bool:
    lx, ly = _to_obb_local(px, py, s)
    return abs(lx) <= s["w"] / 2 and abs(ly) <= s["h"] / 2


def point_in_shape(px: float, py: float, s: dict) -> bool:
    if "r" in s:                return _point_in_circle(px, py, s)
    if s.get("angle") is not None: return _point_in_obb(px, py, s)
    return _point_in_rect(px, py, s)


def _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy) -> bool:
    d1x = bx - ax; d1y = by - ay
    d2x = dx - cx; d2y = dy - cy
    cross = d1x * d2y - d1y * d2x
    if abs(cross) < 1e-10:
        return False
    t = ((cx - ax) * d2y - (cy - ay) * d2x) / cross
    u = ((cx - ax) * d1y - (cy - ay) * d1x) / cross
    return 0 <= t <= 1 and 0 <= u <= 1


def _segment_intersects_rect(ax, ay, bx, by, s) -> bool:
    rx = s["x"]; ry = s["y"]; x2 = rx + s["w"]; y2 = ry + s["h"]
    if ax < rx and bx < rx: return False
    if ax > x2 and bx > x2: return False
    if ay < ry and by < ry: return False
    if ay > y2 and by > y2: return False
    if _point_in_rect(ax, ay, s) or _point_in_rect(bx, by, s): return True
    return (
        _segments_intersect(ax, ay, bx, by, rx, ry, x2, ry) or
        _segments_intersect(ax, ay, bx, by, x2, ry, x2, y2) or
        _segments_intersect(ax, ay, bx, by, x2, y2, rx, y2) or
        _segments_intersect(ax, ay, bx, by, rx, y2, rx, ry)
    )


def _segment_intersects_circle(ax, ay, bx, by, s) -> bool:
    dx = bx - ax; dy = by - ay
    lenSq = dx * dx + dy * dy
    t = 0.0
    if lenSq > 0:
        t = ((s["cx"] - ax) * dx + (s["cy"] - ay) * dy) / lenSq
        t = max(0.0, min(1.0, t))
    qx = ax + t * dx; qy = ay + t * dy
    ex = qx - s["cx"]; ey = qy - s["cy"]
    return ex * ex + ey * ey <= s["r"] * s["r"]


def _segment_intersects_obb(ax, ay, bx, by, s) -> bool:
    ax2, ay2 = _to_obb_local(ax, ay, s)
    bx2, by2 = _to_obb_local(bx, by, s)
    local = {"x": -s["w"] / 2, "y": -s["h"] / 2, "w": s["w"], "h": s["h"]}
    return _segment_intersects_rect(ax2, ay2, bx2, by2, local)


def segment_intersects_shape(ax, ay, bx, by, s) -> bool:
    if "r" in s:                return _segment_intersects_circle(ax, ay, bx, by, s)
    if s.get("angle") is not None: return _segment_intersects_obb(ax, ay, bx, by, s)
    return _segment_intersects_rect(ax, ay, bx, by, s)


def _shape_envelope(s: dict) -> dict:
    """Axis-aligned outer bbox of a shape (identity for AABB)."""
    if "r" in s:
        return {"x": s["cx"] - s["r"], "y": s["cy"] - s["r"], "w": 2 * s["r"], "h": 2 * s["r"]}
    if s.get("angle") is None:
        return s
    rad = math.radians(s["angle"])
    c = abs(math.cos(rad)); n = abs(math.sin(rad))
    ew = s["w"] * c + s["h"] * n
    eh = s["w"] * n + s["h"] * c
    return {"x": s["cx"] - ew / 2, "y": s["cy"] - eh / 2, "w": ew, "h": eh}


# ─── Board ───────────────────────────────────────────────────────────────────

# 16-direction neighbours: cardinal, diagonal, knight's-move (matches JS).
_SQRT2 = math.sqrt(2)
_SQRT5 = math.sqrt(5)
_NEIGHBOURS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, _SQRT2), (1, -1, _SQRT2), (-1, 1, _SQRT2), (1, 1, _SQRT2),
    (-2, -1, _SQRT5), (-2, 1, _SQRT5), (2, -1, _SQRT5), (2, 1, _SQRT5),
    (-1, -2, _SQRT5), (-1, 2, _SQRT5), (1, -2, _SQRT5), (1, 2, _SQRT5),
]


class Board:
    def __init__(self, width_in: float, height_in: float, px_per_in: float = 18.0):
        self.width_in = width_in
        self.height_in = height_in
        self.px_per_in = px_per_in
        self.terrain: list[Terrain] = []
        self.nav_cols = int(math.ceil(width_in / NAV_RES))
        self.nav_rows = int(math.ceil(height_in / NAV_RES))
        self._nav_base: np.ndarray | None = None

    def add_terrain(self, t: Terrain | dict) -> None:
        if isinstance(t, dict):
            t = Terrain(**t)
        self.terrain.append(t)
        self._nav_base = None

    def dist_inches(self, ax, ay, bx, by) -> float:
        return math.hypot(bx - ax, by - ay)

    def clamp(self, x, y, r=0.0) -> tuple[float, float]:
        return (
            max(r, min(self.width_in - r, x)),
            max(r, min(self.height_in - r, y)),
        )

    # ─── LOS ────────────────────────────────────────────────────────────────

    def has_los(self, ax, ay, bx, by) -> bool:
        for t in self.terrain:
            if not t.blocks_los:
                continue
            s = t.shape()
            if point_in_shape(ax, ay, s):  # shooter inside → transparent
                continue
            if point_in_shape(bx, by, s):  # target inside → transparent
                continue
            if segment_intersects_shape(ax, ay, bx, by, s):
                return False
        return True

    # ─── Pathfinding ────────────────────────────────────────────────────────

    def _build_nav_grid(self) -> np.ndarray:
        if self._nav_base is not None:
            return self._nav_base

        grid = np.ones(self.nav_cols * self.nav_rows, dtype=np.float32)

        for t in self.terrain:
            if not t.blocks_move and not t.difficult:
                continue
            s = t.shape()
            env = _shape_envelope(s)
            c0 = int(math.floor(env["x"] / NAV_RES))
            r0 = int(math.floor(env["y"] / NAV_RES))
            c1 = int(math.ceil((env["x"] + env["w"]) / NAV_RES))
            r1 = int(math.ceil((env["y"] + env["h"]) / NAV_RES))
            cost = math.inf if t.blocks_move else DIFFICULT_COST

            for ri in range(r0, r1):
                if ri < 0 or ri >= self.nav_rows:
                    continue
                for ci in range(c0, c1):
                    if ci < 0 or ci >= self.nav_cols:
                        continue
                    cx_pt = (ci + 0.5) * NAV_RES
                    cy_pt = (ri + 0.5) * NAV_RES
                    if not point_in_shape(cx_pt, cy_pt, s):
                        continue
                    grid[ri * self.nav_cols + ci] = cost

        self._nav_base = grid
        return grid

    def flood_fill(self, start_x: float, start_y: float,
                   max_dist: float, is_infantry: bool) -> np.ndarray:
        """Dijkstra flood returning cost-to-reach (Inf = unreachable or past max_dist).

        NAV_RES is already baked into `baseCost` entries in JS; we keep that
        convention — each neighbour's cost is neighbour_weight * NAV_RES,
        multiplied by cell cost if non-infantry."""
        base = self._build_nav_grid()
        n = self.nav_cols * self.nav_rows
        dist = np.full(n, math.inf, dtype=np.float32)

        sc = int(round(start_x / NAV_RES))
        sr = int(round(start_y / NAV_RES))
        if sc < 0 or sc >= self.nav_cols or sr < 0 or sr >= self.nav_rows:
            return dist
        start = sr * self.nav_cols + sc
        dist[start] = 0.0

        heap: list[tuple[float, int]] = [(0.0, start)]

        while heap:
            d, idx = heapq.heappop(heap)
            if d > dist[idx]:
                continue
            if d > max_dist:
                continue
            r = idx // self.nav_cols
            c = idx % self.nav_cols

            for dc, dr, step_weight in _NEIGHBOURS:
                nc = c + dc; nr = r + dr
                if nc < 0 or nc >= self.nav_cols or nr < 0 or nr >= self.nav_rows:
                    continue
                nidx = nr * self.nav_cols + nc
                cell_cost = base[nidx]
                if cell_cost == math.inf:
                    continue

                # Knight's-move: ensure we don't squeeze through walls
                if abs(dc) + abs(dr) > 2:
                    sign_dr = (1 if dr > 0 else -1 if dr < 0 else 0)
                    sign_dc = (1 if dc > 0 else -1 if dc < 0 else 0)
                    mid1 = (r + sign_dr) * self.nav_cols + (c + sign_dc)
                    if abs(dc) > abs(dr):
                        mid2 = r * self.nav_cols + (c + sign_dc)
                    else:
                        mid2 = (r + sign_dr) * self.nav_cols + c
                    if base[mid1] == math.inf or base[mid2] == math.inf:
                        continue

                base_cost = step_weight * NAV_RES
                move_cost = base_cost if is_infantry else base_cost * cell_cost
                new_dist = d + move_cost
                if new_dist < dist[nidx] and new_dist <= max_dist:
                    dist[nidx] = new_dist
                    heapq.heappush(heap, (new_dist, nidx))

        return dist

    def reachable_cells(self, dist_grid: np.ndarray) -> list[tuple[float, float]]:
        cells: list[tuple[float, float]] = []
        for i in range(dist_grid.size):
            if not math.isfinite(dist_grid[i]):
                continue
            r = i // self.nav_cols
            c = i % self.nav_cols
            cells.append((c * NAV_RES, r * NAV_RES))
        return cells

    def can_move_to(self, sx, sy, dx, dy, movement, is_infantry) -> bool:
        grid = self.flood_fill(sx, sy, movement, is_infantry)
        dc = int(round(dx / NAV_RES))
        dr = int(round(dy / NAV_RES))
        if dc < 0 or dc >= self.nav_cols or dr < 0 or dr >= self.nav_rows:
            return False
        return math.isfinite(grid[dr * self.nav_cols + dc])

    def snap_to_grid(self, x: float, y: float) -> tuple[float, float]:
        c = int(round(x / NAV_RES))
        r = int(round(y / NAV_RES))
        c = max(0, min(self.nav_cols - 1, c))
        r = max(0, min(self.nav_rows - 1, r))
        return (c * NAV_RES, r * NAV_RES)

    # ─── Objective helper ──────────────────────────────────────────────────

    def controlling_team(self, obj_x: float, obj_y: float,
                         units, radius: float = 3.0) -> int:
        oc = [0, 0]
        for u in units:
            if u.is_dead:
                continue
            if self.dist_inches(u.x, u.y, obj_x, obj_y) <= radius:
                oc[u.team] += u.objective_control
        if oc[0] == 0 and oc[1] == 0:
            return -1
        if oc[0] == oc[1]:
            return -1
        return 0 if oc[0] > oc[1] else 1


# ─── Default terrain layouts (match what GameScene.js uses) ──────────────────

def default_terrain_portrait_44x60() -> list[Terrain]:
    """Portrait board terrain: mirror the authored layout from GameScene.js.

    The JS scene hand-places ~10 pieces; we mirror a representative set so
    LOS interactions are broadly similar. Training does not need exact
    pixel parity — just sightline-breaking terrain in the middle band
    (y=22..38) per CLAUDE.md."""
    return [
        Terrain(x=18, y=14, w=8, h=4, label="rect_ruins"),
        Terrain(x=28, y=18, w=6, h=6, label="square_bunker"),
        Terrain(x=12, y=30, w=10, h=4, label="rect_dropship", blocks_move=True),
        Terrain(x=30, y=34, w=8, h=6, label="l_shaped_ruins"),
        Terrain(x=22, y=44, w=6, h=4, label="rect_comms_tower"),
        Terrain(cx=10, cy=50, r=3.0, angle=None, label="fuel_silo_lower"),
        Terrain(cx=34, cy=10, r=3.0, angle=None, label="fuel_silo_upper"),
        Terrain(x=8, y=26, w=4, h=4, difficult=True, blocks_move=False,
                blocks_los=False, label="rubble_left"),
        Terrain(x=36, y=38, w=4, h=4, difficult=True, blocks_move=False,
                blocks_los=False, label="rubble_right"),
    ]
