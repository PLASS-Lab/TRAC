"""Supplementary Table S1 entry point: Constructive rates.

Runs `tworead.theory.rates` with the same arguments, e.g.
    python scripts/verify_two_read_bridge.py --output out/rates.json
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_module('tworead.theory.rates', run_name='__main__')
