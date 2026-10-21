#!/usr/bin/env python3
"""Verify the provenance, vector embedding and unchanged v3.2 plotted arrays."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT/'paper/mdpi_math_v3_2'
WORK = ROOT/'work_dirs/mdpi_math_v3_2'
VISUAL = WORK/'visual_revision'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plotted_signature(fig):
    signature = hashlib.sha256()
    for ax in fig.axes:
        for line in ax.lines:
            for values in (line.get_xdata(),line.get_ydata()):
                array = np.asarray(values)
                signature.update(str(array.shape).encode())
                signature.update(array.tobytes())
    return signature.hexdigest()


def original_plot_signature(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.OUT, module.PAPER = ROOT, WORK, PAPER
    signatures=[]
    with patch.object(Figure,'savefig',lambda fig,*a,**kw:signatures.append(plotted_signature(fig))):
        module.main()
    assert len(signatures)==2 and signatures[0]==signatures[1]
    return signatures[0]


def main():
    checks={}
    protocol=json.loads((VISUAL/'protocol.json').read_text())
    manifest=json.loads((VISUAL/'figure_manifest.json').read_text())
    checks['frozen_results_unchanged']=protocol['results_sha256']==digest(WORK/'results.json')
    checks['sampled_families_unchanged']=protocol['families_sha256']==digest(WORK/'sampled_families.npz')
    checks['all_eight_new_assets_match']=all(digest(PAPER/path)==value for path,value in manifest['figures'].items())
    old=original_plot_signature(VISUAL/'pre_visual/build_math_v3_2_figure.py','old_variance_figure')
    new=original_plot_signature(ROOT/'scripts/build_math_v3_2_figure.py','current_variance_figure')
    checks['existing_variance_figure_arrays_unchanged']=old==new
    names=['shared_atom_structure','adaptive_construction','range_embedding_identity',
           'normalization_stability','variance_adaptive']
    fonts={}
    for name in names:
        path=PAPER/'figures'/f'{name}.pdf'
        proc=subprocess.run(['pdffonts',str(path)],capture_output=True,text=True,check=True)
        fonts[name]=proc.stdout
        checks[name+'_valid_serif_embedding']=(not proc.stderr.strip() and 'PagellaFigure' in proc.stdout
            and not any(text in proc.stdout for text in ('Type 3','Sans','Helvetica')))
        images=subprocess.check_output(['pdfimages','-list',str(path)],text=True)
        checks[name+'_vector_graphics']=len(images.strip().splitlines())==2
    output={'passed':all(checks.values()),'checks':checks,'source_data_signature':old,
            'figure_fonts':fonts,'scope':'figure provenance, arrays and PDF embedding; no_new_experiment'}
    (VISUAL/'visual_validation.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='figure_fonts'},indent=2))
    raise SystemExit(0 if output['passed'] else 1)


if __name__=='__main__':
    main()
