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

Suppose you suspect Harrow's Render is over-statted. To test:

1. **Lower its ballistic skill in the Python env** (`threnody_rl/env/factions.py`).
   Change `ballistic_skill=4` to `ballistic_skill=5` for the Render entry.
2. **Run eval against an existing checkpoint** trained on the unmodified
   stats. The opponent's policy hasn't been updated, so this measures how
   much the stat change shifts win rates *without* needing a full retrain.
3. **Compare** the new matchup matrix against a baseline matrix you saved
   earlier (use `--out matrix_after.json` then diff).

For more rigorous results, retrain after the stat change so both sides
adapt — but the quick eval is usually enough to flag obvious imbalances.

(A `tune.py` script that automates this stat-tweak → eval → delta-report
loop is on the wishlist; see `threnody/docs/MULTIPLAYER_PHASE1_PLAN.md`'s
Phase 1.x follow-ups. For now the workflow is manual.)

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
