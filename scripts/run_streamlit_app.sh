#!/usr/bin/env bash
# Run Streamlit SO101 inference app. Bind to 0.0.0.0 for AWS/remote access.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="./lerobot/src:${PYTHONPATH}"
exec uv run streamlit run scripts/app_inference_viz.py --server.address 0.0.0.0 "$@"
