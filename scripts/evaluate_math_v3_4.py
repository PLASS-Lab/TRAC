"""Supplementary Table S1 entry point: Kodak storage and query costs.

Runs `tworead.experiments.kodak` with the same arguments, e.g.
    python scripts/evaluate_math_v3_4.py download|freeze|run|summarize --out out/kodak, or replay --evidence ...
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_module('tworead.experiments.kodak', run_name='__main__')
