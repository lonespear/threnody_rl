# Threnody RL — faction balance harness

A Python port of the Threnody combat engine + a PPO self-play agent, used
to surface faction balance issues before they ship to the live game.

## What this is for

When you change a unit's stats in `threnody_rl/env/factions.py` (mirroring
a change you'd make in the live game's `src/data/<faction>.js`), you can:

1. Train a policy (or reuse an existing one) that plays both sides of every
   faction matchup.
2. Run a round-robin evaluation across all 25 (player faction × enemy
   faction) pairings.
3. Read off win-rate disparities to spot factions that are too strong or
   too weak after the change.

The audit on 2026-04-28 confirmed combat-math, GameState, and faction-stat
parity with the live JS engine. So balance findings here transfer to the
live game.

## Quick start (CPU)

```bash
# One-time setup
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m threnody_rl.tests.test_parity   # ~30s; should pass cleanly

# Train a baseline checkpoint (~6-12 hours CPU; faster on GPU)
python -m threnody_rl.train \
  --total-steps 4000000 \
  --rollout-steps 2048 \
  --minibatch-size 256 \
  --epochs 4 \
  --log-interval 1000 \
  --save-interval 100000 \
  --league-interval 100000

# Evaluate balance: 30 games per matchup × 25 matchups = 750 games
python -m threnody_rl.eval \
  --checkpoint checkpoints/threnody_step_4000000_final.pt \
  --games-per-matchup 30 \
  --opponent self
```

The eval prints a 5×5 matchup matrix (rows = player faction, columns =
enemy faction) with win/loss/draw counts. Diagonal = mirror match.

## Workflow: testing a stat change

Use `tune.py`. It runs a baseline 5×5 matrix, applies your stat overrides
in-memory, runs the same matrix again, and prints the per-matchup delta.
No file edits, no retraining.

```bash
python -m threnody_rl.tune \
  --checkpoint checkpoints/threnody_step_4000000_final.pt \
  --override harrow.render.ballistic_skill=4 \
  --override accord.line_trooper.ranged.attacks=2 \
  --games-per-matchup 30 \
  --out runs/render_nerf.json
```

Override syntax: `<faction>.<unit_slug>.<field>=<value>`, repeated as
many times as needed. Slug = unit's display name lowercased with spaces
replaced by underscores ("Line Trooper" → `line_trooper`, "Render" →
`render`). Field paths support nested weapons (`ranged.attacks`,
`melee.damage`, etc.) and top-level Unit attributes (`movement`,
`toughness`, `weapon_skill`, `ballistic_skill`, `wounds`, `save`,
`nullfield_save`, `objective_control`, `advance_bonus`).

**Why this works without retraining.** Both sides use the same fixed
policy, so the policy-vs-policy strategic equilibrium doesn't shift —
the only thing that changes is the stat block. The win-rate delta
isolates the *mechanical* effect of the stat change. For a deeper test
where the policy adapts to the new stats, retrain afterward — but most
"is X over-statted?" questions answer cleanly with the quick eval.

Output:
- Baseline 5×5 matrix (rows = player faction, cols = enemy faction)
- Modified 5×5 matrix
- Delta table (positive = override helped player faction)
- Per-faction average win rate before/after across all 5 matchups
- Optional JSON dump (`--out path/to/file.json`) for later comparison

## Hardware notes

- Training to 4M steps on a single CPU thread takes 6-12 hours
- A consumer GPU (RTX 3060+) does the same in 1-3 hours
- Each evaluation game is ~5-10 seconds CPU; 750 games for a full matrix is
  ~1-2 hours
- `dashboard.py` runs a Flask server on port 5050 that streams training
  metrics live — useful for catching divergence early

## Hyperparameter tuning history

See the FortyK RL run history in `~/.claude/projects/.../memory/MEMORY.md`
under `# FortyK RL Project` for prior hyperparameter trial results
(HIDDEN=512 + league play + VP-delta reward worked best; entropy < 1.0 is
the canary for premature convergence).

## Parity check

Every change to unit stats, combat math, or game logic in the live JS
engine should be mirrored here within the same release cycle, otherwise
balance findings drift away from real-game behaviour. The audit identifies:

- `threnody_rl/env/factions.py` ↔ `threnody/src/data/<faction>.js`
- `threnody_rl/env/unit.py` ↔ `threnody/src/core/combat.js`
- `threnody_rl/env/game_state.py` ↔ `threnody/src/core/GameState.js`
- `threnody_rl/env/board.py` ↔ `threnody/src/core/Board.js`

Run `python -m threnody_rl.tests.test_parity` after any sync to confirm
combat math still matches.
