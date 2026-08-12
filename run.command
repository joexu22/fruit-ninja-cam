#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
set +e
python -m fruit_ninja_cam 2>&1 | tee /tmp/fruit-ninja-cam.log
code=${PIPESTATUS[0]}
echo ""
echo "Exit code: $code"
echo "Log: /tmp/fruit-ninja-cam.log"
echo "Press Enter to close…"
read -r
