"""Supplementary Table S1 entry point: Geometric boundary audit.

Runs `tworead.theory.geometry_audit` with the same arguments, e.g.
    python scripts/audit_math_v3_4.py --output out/geometry.json
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_module('tworead.theory.geometry_audit', run_name='__main__')
