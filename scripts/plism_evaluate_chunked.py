#!/usr/bin/env python3
"""Run `plismbench evaluate` with the top-k step computed in row chunks.

plismbench's TopkAccuracy builds the full (2n x 2n) cosine matrix and an
int64 argpartition of the same shape (2.1 GB at n = 8,139). With GPU memory
partly held by another tenant this OOMs. Each row's top-k depends only on
that row, so this patch computes the same fp16 cosine rows and the same
argpartition chunk by chunk; everything else (pair list, tile subset,
cosine-similarity metric, aggregation, output files) is plismbench's own code.
Checked against unchunked plismbench output on a completed run (--limit).

Usage:
  python3 scripts/plism_evaluate_chunked.py --extractor uni/vit-small \
      [--metrics-dir outputs/plism/results] [--limit N]
"""
import argparse
from pathlib import Path

import numpy as np
from plismbench.metrics import retrieval
from plismbench.engine import evaluate as ev

CHUNK = 2048


def compute_metric_chunked(self, matrix_a, matrix_b):
    if matrix_a.shape[0] != matrix_b.shape[0]:
        raise ValueError("Number of tiles must match.")
    ncp = self.ncp
    matrix_ab = np.concatenate([matrix_a, matrix_b], axis=0)
    n_tiles = matrix_ab.shape[0] // 2
    if self.use_mixed_precision:
        matrix_ab = matrix_ab.astype(np.float16)
    matrix_ab = ncp.asarray(matrix_ab)
    norm_ab = ncp.linalg.norm(matrix_ab, axis=1, keepdims=True)
    kmax = max(self.k)
    top = []
    for i in range(0, matrix_ab.shape[0], CHUNK):
        rows = matrix_ab[i:i + CHUNK]
        cos = ncp.matmul(rows, matrix_ab.T) / (norm_ab[i:i + CHUNK] * norm_ab.T)
        top.append(ncp.argpartition(-cos, range(1, kmax + 1), axis=1)[:, 1:kmax + 1])
        del cos
    top_kmax_indices_ab = ncp.concatenate(top, axis=0)
    out = []
    for k in self.k:
        idx = top_kmax_indices_ab[:, :k]
        accs = []
        for j, ind in enumerate([idx[:n_tiles], idx[n_tiles:]]):
            other = ncp.arange(n_tiles, 2 * n_tiles) if j == 0 else ncp.arange(0, n_tiles)
            c = ncp.sum(ncp.any(ind == other[:, None], axis=1)) / n_tiles
            accs.append(float(c.get()) if self.device == "gpu" else float(c))
        out.append(sum(accs) / 2)
    return np.array(out)


retrieval.TopkAccuracy.compute_metric = compute_metric_chunked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", required=True)
    ap.add_argument("--features-dir", default="outputs/plism/features")
    ap.add_argument("--metrics-dir", default="outputs/plism/results")
    ap.add_argument("--limit", type=int, default=0,
                    help="validation only: evaluate the first N pairs")
    a = ap.parse_args()
    if a.limit:
        orig = ev.prepare_pairs_dataframe
        ev.prepare_pairs_dataframe = lambda **kw: orig(**kw).iloc[:a.limit]
    ev.compute_metrics(features_root_dir=Path(a.features_dir),
                       metrics_save_dir=Path(a.metrics_dir),
                       extractor=a.extractor, device="gpu")


if __name__ == "__main__":
    main()
