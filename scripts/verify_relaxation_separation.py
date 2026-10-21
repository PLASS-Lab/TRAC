"""Supplementary Table S1 entry point: Relaxation separation.

Runs `tworead.theory.relaxation` with the same arguments, e.g.
    python scripts/verify_relaxation_separation.py --output out/relaxation.json
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_module('tworead.theory.relaxation', run_name='__main__')
