#!/usr/bin/env bash
# Run 3 — stalemate-breaking retrain.
# Resumes from the Run 2 final checkpoint and adds 2M more steps under a
# reward recipe designed to push the policy out of alive-count-tiebreak
# grinding into actual wipeouts. See threnody_rl.md "Path B" for context.
#
# Usage:  ./run_training_run3.sh
#         (then: tail -f training_run3.log)

set -euo pipefail

CHECKPOINT="checkpoints/threnody_step_4001792_final.pt"
LOG="training_run3.log"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found at $CHECKPOINT" >&2
  exit 1
fi

# Kill any existing train.py process to avoid double-launch
if pgrep -fa "threnody_rl.train" >/dev/null; then
  echo "Existing train process found — killing first:"
  pgrep -fa "threnody_rl.train"
  pkill -f "threnody_rl.train" || true
  sleep 1
fi

echo "Launching Run 3 retrain in background. Log: $LOG"
nohup python -m threnody_rl.train \
  --resume "$CHECKPOINT" \
  --total-steps 6000000 \
  --ko-bonus 5.0 \
  --dmg-dealt 0.0 \
  --draw-penalty 30 \
  --rollout-steps 4096 \
  --minibatch-size 256 \
  --epochs 4 \
  --log-interval 5000 \
  --save-interval 100000 \
  --league-interval 100000 \
  --mode DEATHMATCH \
  > "$LOG" 2>&1 &

PID=$!
echo "Started PID $PID"
sleep 3
if kill -0 "$PID" 2>/dev/null; then
  echo "Process $PID is alive."
  echo "Tail the log with:  tail -f $LOG"
else
  echo "ERROR: process $PID died within 3 seconds. Check $LOG for the traceback." >&2
  tail -30 "$LOG" >&2
  exit 1
fi
