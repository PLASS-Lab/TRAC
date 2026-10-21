"""Supplementary Table S1 entry point: Shared-atom constants.

Runs `tworead.theory.constants` with the same arguments, e.g.
    python scripts/verify_improved_constant.py --output out/constants.json
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_module('tworead.theory.constants', run_name='__main__')
