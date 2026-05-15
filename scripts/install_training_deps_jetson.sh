#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/csteinhauer/robot_vision"

cd "${APP_DIR}"

echo "Installing Robot Vision Hugging Face training dependencies..."
python3 -m pip install --user -e ".[train]"

echo "Verifying training dependencies..."
python3 - <<'PY'
from robot_vision.training.hf_vision import check_training_dependencies

status = check_training_dependencies()
print(status)
if not status["ok"]:
    print()
    print("If torchvision is still missing on Jetson, install a torchvision build")
    print("that matches the Jetson's installed PyTorch/L4T version, then rerun this script.")
    raise SystemExit(1)
PY

echo "Training dependencies are installed."
