# Slide-cluster-robust equivalence testing for distilled pathology foundation models

Code release: the distillation, evaluation and statistics pipeline behind a
controlled 36-run teacher/student atlas (12 public pathology foundation models
× 3 ViT student sizes), together with a reusable implementation of
slide-cluster-robust equivalence testing (TOST + slide-cluster bootstrap +
hierarchical FDR).

This repository contains **code and operating instructions only**. No
manuscript text, figures or results are included here. Trained checkpoints,
patch bundles and result JSONs are distributed separately (see
[Data and checkpoints](#data-and-checkpoints)).

---

## What is in here

| Path | Contents |
|---|---|
| `scripts/stats_v4.py` | **Self-contained statistics module.** Slide-cluster bootstrap, TOST equivalence, hierarchical Benjamini–Hochberg. Reusable on any benchmark with a patch→slide mapping. |
| `scripts/distill_*.py` | Distillation training (teacher zoo loading, student models, cosine CLS + patch-token loss). |
| `scripts/eval_*.py` | Equivalence evaluations (CAMELYON16, Kather-MSI, full variance stack, external-model audit). |
| `scripts/extract_*.py`, `scripts/build_*.py` | Patch extraction and HDF5 bundle construction. |
| `scripts/compute_*.py`, `scripts/audit_*.py` | Derived statistics: SE inflation, intra-slide ICC, FLOPs, calibration, slide-split leakage audit. |
| `scripts/sim_*.py` | Calibration simulations for the equivalence test and its FDR control. |
| `scripts/figure_*.py` | Figure generation (matplotlib). |
| `scripts/run_*.sh` | Sweep drivers for the full atlas, evaluation and replicated-seed runs. |
| `configs/` | Multi-source dataset configuration and a smoke-test config. |
| `tests/` | Unit tests (teacher-specific input normalisation). |
| `release/checksums_sha256.txt` | SHA-256 manifest of the released student checkpoints. |

### The statistics module on its own

`scripts/stats_v4.py` has no dependency on the rest of the repository — only
NumPy. If you want the method and not the atlas, this is the file:

```python
from stats_v4 import slide_cluster_bootstrap, tost_equivalence, hierarchical_bh

# `groups` is the slide (or patient) id of every patch — this is what makes
# the interval cluster-robust instead of anti-conservative.
result = tost_equivalence(t_correct, s_correct, groups, margin=0.05,
                          n_boot=2000, seed=42)
```

Run its tests with:

```bash
python3 -m pytest scripts/test_stats_v4.py -q
```

---

## Installation

Python 3.10+ and a CUDA-capable GPU are assumed for the training and
feature-extraction stages. The statistics and simulation scripts run on CPU.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`openslide-python` needs the OpenSlide C library present first:

```bash
# Debian/Ubuntu
sudo apt-get install openslide-tools
```

Some teacher encoders are gated on Hugging Face and require you to accept
their licence and authenticate before the loader can fetch them:

```bash
huggingface-cli login
```

---

## Configuration

Every path in this repository is a **placeholder** of the form
`/path/to/...` — substitute your own before running anything.

```bash
cp configs/paths.env.example configs/paths.env
$EDITOR configs/paths.env      # point at your WSI, cache and project dirs
source configs/paths.env
```

`configs/multi_source_v1.json` declares the training patch sources; edit the
`path` / `root` fields to your local bundles. `configs/smoke.json` is a small
config for verifying the pipeline runs end to end before committing GPU time.

Placeholders you will encounter:

| Placeholder | Meaning |
|---|---|
| `/path/to/wsi_hl` | This repository's working directory |
| `/path/to/wsi_datasets` | Raw whole-slide image cohorts |
| `/path/to/cache` | Fast local cache for HDF5 patch bundles and features |

---

## Pipeline

The stages below are the order they were run in. Each is independently
resumable, and evaluation stages skip runs whose report already exists.

### 1. Acquire slides and build patch bundles

```bash
python3 scripts/download_camelyon16.py --out /path/to/wsi_datasets/camelyon16

python3 scripts/downloads/download_tcga.py --list      # cohort → project_id map
python3 scripts/downloads/download_tcga.py --dry-run   # file count and total size
python3 scripts/downloads/download_tcga.py --only brca
```

The TCGA downloader shells out to `gdc-client`, which must be on your `PATH`.
Downloading every cohort is ~3.7 TB, so start with `--dry-run`.

Tile the slides into HDF5 bundles. For CAMELYON16 the tumour label must come
from the **lesion annotation polygon**, not the filename — filename-level
labelling puts most tiles of a tumour slide in the wrong class and drives the
tile-level probe to near chance:

```bash
python3 scripts/extract_c16_annotated.py \
    --wsi_dir /path/to/wsi_datasets/camelyon16 \
    --ann_dir data/c16_annotations \
    --out /path/to/cache/patches/patches_c16_annotated.h5

python3 scripts/extract_patches.py --help          # generic extractor
python3 scripts/build_kather_msi_h5.py --help      # second cohort
```

### 2. Distil the atlas

12 teachers × 3 student sizes = 36 runs, shared recipe, single seed:

```bash
bash scripts/run_v4_full.sh
```

Replicated-seed subset (for the variance decomposition):

```bash
bash scripts/run_v4_seed2.sh
```

Smoke-test a single pair first:

```bash
python3 scripts/distill_v4.py --config configs/smoke.json --epochs 1
```

### 3. Verify the splits before trusting any result

```bash
python3 scripts/audit_slide_splits_slidelevel.py
```

This must report zero cross-split slide overlaps. Tile-level leakage between
train and test is the failure mode that silently inflates every downstream
number, so treat a non-zero count as a hard stop.

### 4. Downstream evaluation

```bash
bash scripts/run_v4_eval_all.sh outputs/v4_full
```

### 5. Equivalence certification

The headline analysis. Re-extracts features from the frozen encoders, builds
its own slide-stratified split with enough test slides for cluster-robust
inference, and runs the TOST:

```bash
python3 scripts/eval_c16_equivalence.py \
    --h5 /path/to/cache/patches/patches_c16_annotated.h5 \
    --test_frac 0.5 --out outputs/v4_full/c16_equiv_annotated

python3 scripts/eval_kather_msi_equivalence.py    # second cohort, patient clusters
python3 scripts/eval_c16_fullstack.py             # probe + seed + cluster variance
```

Derived quantities:

```bash
python3 scripts/compute_cluster_se_inflation.py   # cluster vs naive SE
python3 scripts/compute_intra_slide_icc.py        # intra-slide correlation
python3 scripts/variance_decomposition.py
```

### 6. Calibration simulations

These need no data and reproduce in seconds — the quickest way to see what the
cluster-robust test buys you:

```bash
python3 scripts/sim_equivalence_calibration.py
python3 scripts/sim_fdr_equivalence.py
```

### 7. Figures

```bash
mkdir -p paper/figures        # figure scripts write here by default
python3 scripts/figure_equivalence.py
python3 scripts/figure_center_plism.py
python3 scripts/figure_convergence_curves.py
```

---

## Data and checkpoints

Not in this repository. Restricted sources must be obtained from their
original custodians:

| Source | Where |
|---|---|
| CAMELYON16 WSIs | https://camelyon17.grand-challenge.org/Data/ |
| TCGA-BRCA WSIs | GDC Data Portal, project `TCGA-BRCA` |
| NCT-CRC-HE-100K, CRC-VAL-HE-7K | Zenodo release of the original publication |
| Kather-MSI, PanNuke | original Zenodo releases |
| PLISM | consumed via the `plismbench` protocol |

Student checkpoints, per-run splits, training logs and evaluation JSONs are
archived separately. `release/checksums_sha256.txt` is the SHA-256 manifest for
that archive — verify a download with:

```bash
sha256sum -c release/checksums_sha256.txt
```

Teacher weights are **not** redistributed here; they are fetched from their
original distribution channels by the loader in `scripts/distill_wsi_model.py`.

---

## Reproducibility notes

- All runs use `seed = 42` unless stated otherwise; the replicated-seed subset
  uses `seed = 2025`.
- Splits are slide-disjoint and deterministic given the seed; each run stores
  its own `splits.npz` next to its checkpoint.
- Two teacher families (`hibou-b`, `hibou-l`) require a teacher-specific input
  normalisation. `tests/test_hibou_normalization.py` guards this; runs made
  before the fix are not comparable to the rest of the atlas.
- The equivalence evaluation is strictly post-hoc on frozen encoders — it
  retrains nothing and touches no distillation artifact.

---

## Licence

CC BY-NC 4.0 — see `LICENSE`. Note that this permits non-commercial use only;
some journals and funders require a permissive licence (MIT/Apache-2.0) for
code released alongside a publication, so check your target venue's policy.

Pretrained teacher weights are not redistributed here and remain under their
own licences.
