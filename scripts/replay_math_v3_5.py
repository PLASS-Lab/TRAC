"""Supplementary Table S1 entry point: portable native replay.

Runs `tworead.experiments.depth replay` on the in-repository evidence, e.g.
    python scripts/replay_math_v3_5.py --output out/depth_replay.json
Rebuild the dataset cases first (python -m tworead.cases build ...).
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if '--evidence' not in sys.argv:
    sys.argv += ['--evidence', str(ROOT/'paper/mdpi_math_v4/supplementary/evidence_v3_5')]
sys.argv[1:1] = ['replay']
runpy.run_module('tworead.experiments.depth', run_name='__main__')
