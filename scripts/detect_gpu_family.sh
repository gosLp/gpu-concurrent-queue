#!/usr/bin/env bash
set -euo pipefail

info="$(rocminfo 2>/dev/null || true)"

if echo "$info" | grep -Eqi 'MI300|gfx942'; then
  echo "mi300a"
elif echo "$info" | grep -Eqi 'MI210|gfx90a'; then
  echo "mi210"
else
  echo "unknown"
fi