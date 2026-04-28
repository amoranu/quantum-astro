#!/usr/bin/env bash
# ============================================================
# Project Pitru-Maraka 2.0 — WSL GPU launcher
# Run from Windows:  wsl -d Ubuntu -- bash quantum-astro/run_wsl.sh
# Or from inside WSL: bash run_wsl.sh
# ============================================================

set -e

VENV="$HOME/.venvs/quantum-astro"
# Windows path → WSL path for the project
PROJECT="/mnt/c/Users/ravii/.gemini/antigravity/playground/quantum-astro"
DATA="$PROJECT/data/candidates_post1970.json"

source "$VENV/bin/activate"

echo "=== Python: $(python3 --version) ==="
echo "=== Device check ==="
python3 -c "
import pennylane as qml, torch
dev = qml.device('lightning.gpu', wires=4)
print('lightning.gpu: OK')
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')
"

cd "$PROJECT"
# -u: unbuffered stdout/stderr so progress is visible in real time
export PYTHONUNBUFFERED=1
python3 -u main.py --data "$DATA" "$@"
