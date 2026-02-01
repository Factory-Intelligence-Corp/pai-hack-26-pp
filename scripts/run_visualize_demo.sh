#!/usr/bin/env bash
# Visualize a LeRobot dataset under ~/data using lerobot-dataset-viz.
# Usage: ./scripts/run_visualize_demo.sh [dataset_name] [episode_index]
# Example: ./scripts/run_visualize_demo.sh so101_bench_real_2_v2.1 0

set -e
DATA_ROOT="${DATA_ROOT:-$HOME/data}"
REPO_ID="${1:-so101_bench_real_2_v2.1}"
EPISODE_INDEX="${2:-0}"

if [[ ! -d "$DATA_ROOT/$REPO_ID" ]]; then
  echo "Dataset not found: $DATA_ROOT/$REPO_ID"
  echo "Place a LeRobot v3 dataset at $DATA_ROOT/<dataset_name> (e.g. $DATA_ROOT/so101_bench_real_2_v2.1) then run again."
  exit 1
fi

exec lerobot-dataset-viz \
  --repo-id "$REPO_ID" \
  --root "$DATA_ROOT" \
  --mode local \
  --episode-index "$EPISODE_INDEX"
