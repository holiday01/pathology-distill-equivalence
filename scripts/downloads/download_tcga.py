#!/usr/bin/env python3
"""
下載 TCGA Diagnostic Slide (FFPE, SVS) 給 6 cohorts。

流程：
  1. 用 GDC API 對每個 cohort 查 open-access SVS Diagnostic Slide
  2. 產出 per-cohort manifest.txt
  3. 用 gdc-client 並行下載到 /path/to/wsi_datasets/tcga/<cohort>/

用法：
  python download_tcga.py --list                 # 顯示 cohort → project_id 對應
  python download_tcga.py --dry-run              # 只查檔案數與總大小
  python download_tcga.py --only brca            # 只下單一 cohort
  python download_tcga.py --only brca,prad       # 下多個
  python download_tcga.py                        # 全下（~3.7 TB）

Manifest 存在 /path/to/wsi_datasets/tcga/<cohort>/manifest.txt。
Log 存在 /path/to/wsi_datasets/tcga/_logs/<cohort>.log。
"""
import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import os
BASE = Path(os.environ.get("WSI_TCGA_DIR", "/path/to/wsi_datasets/tcga"))
LOGDIR = BASE / "_logs"
GDC_API = "https://api.gdc.cancer.gov/files"

# cohort key → 對應 GDC project_id 列表
COHORTS = {
    "brca":      ["TCGA-BRCA"],
    "coadread":  ["TCGA-COAD", "TCGA-READ"],
    "prad":      ["TCGA-PRAD"],
    "lung":      ["TCGA-LUAD", "TCGA-LUSC"],
    "brain":     ["TCGA-GBM", "TCGA-LGG"],
    "lihc":      ["TCGA-LIHC"],
}


def log(name: str, msg: str):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(f"[{name}] {line}", flush=True)
    with open(LOGDIR / f"{name}.log", "a") as f:
        f.write(line + "\n")


def fmt_bytes(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def query_manifest(cohort: str, project_ids: list[str]) -> list[dict]:
    """查 GDC API 拿 open-access Diagnostic Slide (SVS) 的 file records。"""
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": project_ids}},
            {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
            {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }
    fields = ["file_id", "file_name", "file_size", "md5sum", "state"]
    body = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": "20000",
    }
    data = "&".join(f"{k}={urllib.request.quote(str(v))}" for k, v in body.items()).encode()
    req = urllib.request.Request(GDC_API, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        res = json.loads(r.read())
    hits = res.get("data", {}).get("hits", [])
    return hits


def write_manifest(cohort_dir: Path, hits: list[dict]) -> Path:
    """寫成 gdc-client 需要的 TSV manifest 格式。"""
    cohort_dir.mkdir(parents=True, exist_ok=True)
    manifest = cohort_dir / "manifest.txt"
    with open(manifest, "w") as f:
        f.write("id\tfilename\tmd5\tsize\tstate\n")
        for h in hits:
            f.write(f"{h['file_id']}\t{h['file_name']}\t{h['md5sum']}\t{h['file_size']}\t{h['state']}\n")
    return manifest


def gdc_download(cohort: str, manifest: Path, cohort_dir: Path, n_procs: int = 8) -> bool:
    """呼叫 gdc-client download。"""
    cmd = [
        "gdc-client", "download",
        "-m", str(manifest),
        "-d", str(cohort_dir),
        "--n-processes", str(n_procs),
        "--retry-amount", "5",
        "--wait-time", "10",
        "--log-file", str(LOGDIR / f"{cohort}.gdc.log"),
    ]
    logfile = LOGDIR / f"{cohort}.log"
    log(cohort, f"$ {' '.join(shlex.quote(c) for c in cmd)}")
    with open(logfile, "a") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode == 0:
        log(cohort, f"gdc-client done (rc=0)")
        return True
    log(cohort, f"gdc-client FAIL (rc={r.returncode})")
    return False


def run_cohort(cohort: str, project_ids: list[str], dry_run: bool, n_procs: int):
    cohort_dir = BASE / cohort
    cohort_dir.mkdir(parents=True, exist_ok=True)
    log(cohort, f"query GDC: {project_ids}")
    hits = query_manifest(cohort, project_ids)
    total = sum(int(h["file_size"]) for h in hits)
    log(cohort, f"{len(hits)} files, total {fmt_bytes(total)}")
    if not hits:
        log(cohort, "no files; skip")
        return
    manifest = write_manifest(cohort_dir, hits)
    log(cohort, f"manifest → {manifest}")
    if dry_run:
        return
    t0 = time.time()
    gdc_download(cohort, manifest, cohort_dir, n_procs=n_procs)
    log(cohort, f"elapsed {(time.time()-t0)/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="列出 cohort 對應")
    ap.add_argument("--dry-run", action="store_true", help="只查大小、寫 manifest，不下載")
    ap.add_argument("--only", help="只跑特定 cohort，逗號分隔（如 brca,prad）")
    ap.add_argument("--exclude", help="排除特定 cohort")
    ap.add_argument("--n-procs", type=int, default=8, help="gdc-client 並行 process 數")
    args = ap.parse_args()

    if args.list:
        for k, v in COHORTS.items():
            print(f"  {k:10s} → {', '.join(v)}")
        return

    if not shutil.which("gdc-client"):
        print("ERROR: gdc-client 找不到（檢查 PATH）", file=sys.stderr)
        sys.exit(2)

    targets = list(COHORTS.keys())
    if args.only:
        keep = {k.strip().lower() for k in args.only.split(",")}
        targets = [c for c in targets if c in keep]
    if args.exclude:
        drop = {k.strip().lower() for k in args.exclude.split(",")}
        targets = [c for c in targets if c not in drop]

    log("main", f"cohorts: {targets}  dry_run={args.dry_run}")
    for c in targets:
        run_cohort(c, COHORTS[c], dry_run=args.dry_run, n_procs=args.n_procs)
    log("main", "all done")


if __name__ == "__main__":
    main()
