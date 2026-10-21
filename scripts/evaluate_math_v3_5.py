"""Supplementary Table S1 entry point: Native outputs and matched costs.

Runs `tworead.experiments.depth` with the same arguments, e.g.
    python scripts/evaluate_math_v3_5.py test|freeze|predict|audit|time|summary --out out/depth
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_module('tworead.experiments.depth', run_name='__main__')
