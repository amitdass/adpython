#!/usr/bin/env bash
# DEPRECATED: amplified landmark warps look bad on real faces.
# Use ./run_macos_real_wide.sh instead (real photos only from full wcc1bex tree).
set -euo pipefail

echo "============================================================"
echo "STOPPED: visible warp synthesis is deprecated."
echo
echo "Large 2D warps distort faces (what you saw). That approach"
echo "will not produce usable new angles from aligned crops."
echo
echo "Recommended next step — real-only from your full tree:"
echo "  ./run_macos_real_wide.sh"
echo
echo "Optional cleanup of bad outputs:"
echo "  rm -rf \"/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned/saved_visible\""
echo "============================================================"
exit 1
