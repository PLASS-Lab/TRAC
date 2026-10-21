"""Supplementary Table S1 entry point: Variance and normalization (NYUv2).

Runs `tworead.experiments.variance` with the same arguments, e.g.
    python scripts/evaluate_math_v3_2.py freeze|run|summarize --out out/nyu
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_module('tworead.experiments.variance', run_name='__main__')
