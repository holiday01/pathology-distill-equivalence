#!/usr/bin/env python3
"""Headline TOST with the CAMELYON16 test slides that were also in the
distillation pool removed.

The distillation pool's CAMELYON16 source (patches_camelyon16_full.h5) holds
tiles of 23 training-set slides (normal_001-012, tumor_001-011); 14 of them
fall in the 135-slide test half of the headline evaluation. Students matched
teacher features on those tiles (no labels). This re-runs the slide-cluster
TOST on the remaining test slides, from the per-pair scores saved by
eval_c16_equivalence.py --save_scores.
"""
import sys, glob, os, json, statistics as st
import numpy as np
sys.path.insert(0, '/path/to/wsi_hl/scripts')
from eval_c16_equivalence import auc_cluster_tost
from paper_cohort import keep_teacher
names = [os.path.basename(x) for x in sorted(glob.glob('/path/to/wsi_datasets/camelyon16/*.tif'))]
pool = {'normal_%03d.tif' % i for i in range(1, 13)} | {'tumor_%03d.tif' % i for i in range(1, 12)}
bad = {i for i, n in enumerate(names) if n in pool}
rows = []
for f in sorted(glob.glob('/path/to/wsi_hl/outputs/v4_full/c16_270_rerun/scores/*__linear.npz')):
    t, s, _ = os.path.basename(f).split('__')
    if not keep_teacher(t): continue
    z = np.load(f); g = z['groups'].astype(int)
    keep = ~np.isin(g, list(bad))
    full = auc_cluster_tost(z['y'], z['t'], z['s'], g, margin=0.05)
    sub = auc_cluster_tost(z['y'][keep], z['t'][keep], z['s'][keep], g[keep], margin=0.05)
    rows.append(dict(teacher=t, student=s, d_full=full['d'], equiv_full=full['equiv'],
                     d_sub=sub['d'], lo_sub=sub['lo'], hi_sub=sub['hi'], equiv_sub=sub['equiv'],
                     n_slides_sub=sub['n_slides'], infl_sub=sub['inflation']))
    print(t, s, round(full['d'], 4), full['equiv'], '->', round(sub['d'], 4), sub['equiv'], flush=True)
print('equiv full', sum(r['equiv_full'] for r in rows), 'equiv excl-pool', sum(r['equiv_sub'] for r in rows), 'of', len(rows))
print('median infl excl', st.median(r['infl_sub'] for r in rows))
json.dump({'n_pool_slides': len(pool), 'n_excluded_test_slides': len({int(x) for x in np.unique(z['groups'])} & bad), 'rows': rows}, open('/path/to/wsi_hl/outputs/v4_full/c16_270/leak_sensitivity.json', 'w'), indent=1)
