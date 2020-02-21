#!/usr/bin/bash
# Run this to do all of the preprocessing.
# Ensure that all data is already loaded to $XC_RAW/N5CC3E
# This will take some time. Relax.
# Output will be written to $XC_PROCESSED/N5CC3E

THIS_SCRIPT_DIR="$XC_SCRIPTS/N5CC3E"

bash "$THIS_SCRIPT_DIR/phasing_cleanup.sh"
bash "$THIS_SCRIPT_DIR/haploblock_cleanup.sh"
bash "$THIS_SCRIPT_DIR/cellSNP_cleanup.sh"
python "$THIS_SCRIPT_DIR/save_as_processed.py"

