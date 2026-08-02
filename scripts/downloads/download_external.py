#!/usr/bin/env python3
"""
統一下載 non-TCGA pathology 資料庫。

支援：
  - 直接 HTTP (Zenodo, HuggingFace, Warwick)
  - AWS S3 public bucket (CAMELYON17)
  - Kaggle datasets
  - Google Drive folders (gdown)

用法：
  python download_external.py --list                        # 列出所有資料庫
  python download_external.py --only nct_crc,pannuke,pcam   # 只下特定
  python download_external.py --exclude camelyon17          # 除某個外全下
  python download_external.py                               # 全下（預設）

所有資料存到 ~/wsi_hl/data/external/<name>/，log 在 _logs/。
"""
import argparse
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path.home() / "wsi_hl" / "data" / "external"
LOGDIR = BASE / "_logs"
HAS_ARIA2 = shutil.which("aria2c") is not None

# -----------------------------------------------------------------------------
# 下載工具
# -----------------------------------------------------------------------------

def log(name: str, msg: str):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(f"[{name}] {line}", flush=True)
    with open(LOGDIR / f"{name}.log", "a") as f:
        f.write(line + "\n")


def fmt_bytes(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def _aria2_download(url: str, dst: Path, name: str, connections: int = 8) -> bool:
    """Multi-connection HTTP via aria2c。支援續傳、自動重試。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 清掉舊的 urllib .part（aria2 格式不相容；如果想保留請先手動 rename）
    part = dst.with_suffix(dst.suffix + ".part")
    if part.exists() and not dst.exists():
        log(name, f"rename legacy {part.name} → {dst.name}（供 aria2 續傳）")
        part.rename(dst)

    log(name, f"aria2 start {dst.name} (-x{connections} -s{connections})")
    cmd = [
        "aria2c",
        f"-x{connections}", f"-s{connections}",
        "-c", "--auto-file-renaming=false", "--allow-overwrite=true",
        "--max-tries=5", "--retry-wait=10", "--timeout=60",
        "--console-log-level=warn", "--summary-interval=30",
        "--show-console-readout=false",
        "-d", str(dst.parent), "-o", dst.name,
        url,
    ]
    logfile = LOGDIR / f"{name}.log"
    LOGDIR.mkdir(parents=True, exist_ok=True)
    with open(logfile, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] $ {' '.join(shlex.quote(c) for c in cmd)}\n")
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode == 0 and dst.exists():
        log(name, f"done {dst.name} → {fmt_bytes(dst.stat().st_size)}")
        return True
    log(name, f"FAIL {dst.name} (aria2 rc={r.returncode})")
    return False


def _http_download_urllib(url: str, dst: Path, name: str, expected_size: int = 0):
    """Fallback：純 Python 單執行緒續傳。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")

    existing = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url)
    if existing > 0:
        req.add_header("Range", f"bytes={existing}-")

    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            total = int(r.headers.get("Content-Length", 0)) + existing
            log(name, f"start {dst.name} ({fmt_bytes(total)})  resume={fmt_bytes(existing)}")
            last = time.time()
            with open(part, "ab") as f:
                while True:
                    buf = r.read(1024 * 1024)
                    if not buf:
                        break
                    f.write(buf)
                    existing += len(buf)
                    if time.time() - last > 15:
                        pct = 100 * existing / total if total else 0
                        log(name, f"  ...{dst.name} {pct:.1f}% ({fmt_bytes(existing)}/{fmt_bytes(total)})")
                        last = time.time()
        part.rename(dst)
        log(name, f"done {dst.name} → {fmt_bytes(dst.stat().st_size)}")
        return True
    except Exception as e:
        log(name, f"FAIL {dst.name}: {e}")
        return False


def http_download(url: str, dst: Path, name: str, expected_size: int = 0, connections: int = 8):
    """Resume-capable HTTP GET。優先用 aria2c 多連線，否則 fallback 到 urllib。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if expected_size == 0 or dst.stat().st_size == expected_size:
            log(name, f"skip {dst.name}（已存在）")
            return True

    if HAS_ARIA2:
        return _aria2_download(url, dst, name, connections=connections)
    return _http_download_urllib(url, dst, name, expected_size)


def aria2_batch(items, out_dir: Path, name: str, jobs: int = 8, connections: int = 4) -> bool:
    """批次下載大量檔案。items = [(url, rel_path, expected_size), ...]。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    remaining = []
    for url, rel, size in items:
        dst = out_dir / rel
        if dst.exists() and (size == 0 or dst.stat().st_size == size):
            continue
        remaining.append((url, rel, size))
    if not remaining:
        log(name, f"all {len(items)} files present, skipping")
        return True
    if not HAS_ARIA2:
        log(name, f"aria2c not found, falling back to urllib ThreadPool ({len(remaining)} files)")
        ok = True
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(_http_download_urllib, u, out_dir / r, name, s) for u, r, s in remaining]
            for f in as_completed(futs):
                ok &= f.result()
        return ok

    input_file = LOGDIR / f"{name}_aria2_input.txt"
    with open(input_file, "w") as f:
        for url, rel, _ in remaining:
            f.write(f"{url}\n")
            f.write(f"  dir={out_dir}\n")
            f.write(f"  out={rel}\n")
    log(name, f"aria2 batch: {len(remaining)}/{len(items)} files  -j{jobs} -x{connections}")
    cmd = [
        "aria2c", "-i", str(input_file),
        f"-j{jobs}", f"-x{connections}", f"-s{connections}",
        "-c", "--auto-file-renaming=false", "--allow-overwrite=true",
        "--max-tries=5", "--retry-wait=10", "--timeout=60",
        "--console-log-level=warn", "--summary-interval=60",
        "--show-console-readout=false",
    ]
    logfile = LOGDIR / f"{name}.log"
    with open(logfile, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] $ {' '.join(shlex.quote(c) for c in cmd)}\n")
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    log(name, f"aria2 batch rc={r.returncode}")
    return r.returncode == 0


def run_cmd(cmd: list, name: str, cwd: Path = None) -> bool:
    log(name, "$ " + " ".join(shlex.quote(c) for c in cmd))
    try:
        r = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)
        if r.stdout:
            log(name, r.stdout[-500:])
        if r.returncode != 0:
            log(name, f"stderr: {r.stderr[-500:]}")
            return False
        return True
    except Exception as e:
        log(name, f"FAIL: {e}")
        return False


def list_s3_prefix(bucket_url: str, prefix: str):
    items, token = [], None
    while True:
        url = f"{bucket_url}/?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        root = ET.fromstring(urllib.request.urlopen(url, timeout=30).read())
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for c in root.findall("s3:Contents", ns):
            items.append((c.find("s3:Key", ns).text, int(c.find("s3:Size", ns).text)))
        trunc = root.find("s3:IsTruncated", ns)
        if trunc is not None and trunc.text == "true":
            token = root.find("s3:NextContinuationToken", ns).text
        else:
            break
    return items


# -----------------------------------------------------------------------------
# 每個資料庫的下載函式
# -----------------------------------------------------------------------------

def dl_nct_crc():
    """NCT-CRC-HE-100K（大腸 9 類組織 patch，~11.7 GB）"""
    name = "nct_crc"
    urls = [
        ("https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip", "NCT-CRC-HE-100K.zip"),
        ("https://zenodo.org/record/1214456/files/CRC-VAL-HE-7K.zip", "CRC-VAL-HE-7K.zip"),
    ]
    ok = True
    for url, fn in urls:
        ok &= http_download(url, BASE / name / fn, name)
    return ok


def dl_kather_msi():
    """Kather MSI/MSS（大腸+胃，FFPE，~47 GB）"""
    name = "kather_msi"
    files = [
        "CRC_DX_TRAIN_MSS.zip", "CRC_DX_TRAIN_MSIMUT.zip",
        "CRC_DX_TEST_MSS.zip", "CRC_DX_TEST_MSIMUT.zip",
        "STAD_TRAIN_MSS.zip", "STAD_TRAIN_MSIMUT.zip",
        "STAD_TEST_MSS.zip", "STAD_TEST_MSIMUT.zip",
    ]
    ok = True
    for fn in files:
        url = f"https://zenodo.org/api/records/2530835/files/{fn}/content"
        ok &= http_download(url, BASE / name / fn, name)
    return ok


def dl_bach():
    """BACH 2018（乳癌 4 類 + ROI，~30 GB）"""
    name = "bach"
    url = "https://zenodo.org/record/3632035/files/ICIAR2018_BACH_Challenge.zip"
    return http_download(url, BASE / name / "ICIAR2018_BACH_Challenge.zip", name)


def dl_pannuke():
    """PanNuke（19 器官 nuclei，Warwick 原始 or HF parquet）"""
    name = "pannuke"
    ok = True
    for i in (1, 2, 3):
        url = f"https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_{i}.zip"
        ok &= http_download(url, BASE / name / f"fold_{i}.zip", name)
    return ok


def dl_pcam():
    """PatchCamelyon（乳癌 327680 patches，HF parquet mirror）"""
    name = "pcam"
    base_url = "https://huggingface.co/datasets/1aurent/PatchCamelyon/resolve/main/data"
    files = []
    # train: 13 shards
    for i in range(13):
        files.append(f"train-{i:05d}-of-00013.parquet")
    # valid: 2 shards
    for i in range(2):
        files.append(f"validation-{i:05d}-of-00002.parquet")
    # test: 2 shards
    for i in range(2):
        files.append(f"test-{i:05d}-of-00002.parquet")
    ok = True
    for fn in files:
        ok &= http_download(f"{base_url}/{fn}", BASE / name / fn, name)
    return ok


def dl_camelyon17():
    """CAMELYON17（乳癌 ~1000 WSI + mask，~2.8 TB）"""
    name = "camelyon17"
    bucket = "https://camelyon-dataset.s3.amazonaws.com"
    out_dir = BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)

    log(name, "listing s3://camelyon-dataset/CAMELYON17/ ...")
    items = list_s3_prefix(bucket, "CAMELYON17/")
    total = sum(s for _, s in items)
    log(name, f"found {len(items)} files, {fmt_bytes(total)}")

    batch = []
    for key, size in items:
        rel = key[len("CAMELYON17/"):]
        if not rel:
            continue
        url = f"{bucket}/{urllib.parse.quote(key)}"
        batch.append((url, rel, size))
    # S3 per-connection 已很快，-x 4 足夠，-j 8 並行多檔案
    return aria2_batch(batch, out_dir, name, jobs=8, connections=4)


def dl_kaggle(dataset: str, name: str):
    """通用 kaggle datasets download"""
    out = BASE / name
    out.mkdir(parents=True, exist_ok=True)
    # 若已有大量檔案，跳過
    existing = list(out.rglob("*"))
    if len([p for p in existing if p.is_file() and p.stat().st_size > 1024]) > 10:
        log(name, f"skip kaggle:{dataset}（已有資料）")
        return True
    ok = run_cmd(["kaggle", "datasets", "download", "-d", dataset, "-p", str(out), "--unzip"], name)
    return ok


def dl_panda():
    """PANDA 前列腺 ISUP grading（11K WSI，~350 GB）"""
    name = "panda"
    out = BASE / name
    out.mkdir(parents=True, exist_ok=True)
    # competitions 不是 dataset，用 competitions download
    return run_cmd(
        ["kaggle", "competitions", "download", "-c", "prostate-cancer-grade-assessment", "-p", str(out)],
        name,
    )


def dl_monuseg():
    return dl_kaggle("tuanledinh/monuseg2018", "monuseg")


def dl_glas():
    return dl_kaggle("sani84/glasmiccai2015-gland-segmentation", "glas")


def dl_lizard():
    return dl_kaggle("aadimator/lizard-dataset", "lizard")


def dl_conic():
    return dl_kaggle("aadimator/conic-challenge-dataset", "conic")


def dl_bcss():
    """BCSS（乳癌 region mask，GDrive folder，~5 GB）"""
    name = "bcss"
    out = BASE / name
    out.mkdir(parents=True, exist_ok=True)
    folder_url = "https://drive.google.com/drive/folders/1zqbdkQF8i5cEmZOGmbdQm-EP8dRYtvss"
    return run_cmd(["gdown", "--folder", folder_url, "-O", str(out)], name)


def dl_nucls():
    """NuCLS（乳癌 nuclei，GDrive folder，~2 GB）"""
    name = "nucls"
    out = BASE / name
    out.mkdir(parents=True, exist_ok=True)
    # corrected single-rater
    folder_url = "https://drive.google.com/drive/folders/1eGlF9Dgu3WMEik4fqj0wJ13LKVufsfZ0"
    return run_cmd(["gdown", "--folder", folder_url, "-O", str(out)], name)


def dl_crag():
    """CRAG 大腸腺體（GitHub COCO 轉換版）"""
    name = "crag"
    out = BASE / name
    out.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/XiaoyuZHK/CRAG-Dataset_Aug_ToCOCO/archive/refs/heads/main.zip"
    return http_download(url, out / "CRAG_COCO.zip", name)


# -----------------------------------------------------------------------------
# 註冊表
# -----------------------------------------------------------------------------

DATASETS = {
    # 免註冊、直接 HTTP
    "nct_crc":     (dl_nct_crc,     "NCT-CRC 大腸組織分類",       "~12 GB"),
    "kather_msi":  (dl_kather_msi,  "Kather MSI/MSS 大腸+胃",      "~47 GB"),
    "bach":        (dl_bach,        "BACH 2018 乳癌",              "~30 GB"),
    "pannuke":     (dl_pannuke,     "PanNuke 19 器官 nuclei",      "~1.6 GB"),
    "pcam":        (dl_pcam,        "PatchCamelyon",               "~7 GB"),
    "camelyon17":  (dl_camelyon17,  "CAMELYON17 乳癌",             "~2.8 TB"),
    # Kaggle
    "panda":       (dl_panda,       "PANDA 前列腺 ISUP",           "~350 GB"),
    "monuseg":     (dl_monuseg,     "MoNuSeg 細胞核",              "~500 MB"),
    "glas":        (dl_glas,        "GlaS 大腸腺體",               "~200 MB"),
    "lizard":      (dl_lizard,      "Lizard 大腸 nuclei",          "~3 GB"),
    "conic":       (dl_conic,       "CoNIC 大腸 nuclei challenge", "~1.3 GB"),
    # GDrive / GitHub
    "bcss":        (dl_bcss,        "BCSS 乳癌 region mask",       "~5 GB"),
    "nucls":       (dl_nucls,       "NuCLS 乳癌 nuclei",           "~2 GB"),
    "crag":        (dl_crag,        "CRAG 大腸腺體 (COCO 版)",     "~1.5 GB"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="列出所有資料庫")
    ap.add_argument("--only", default="", help="逗號分隔的資料庫名稱")
    ap.add_argument("--exclude", default="", help="逗號分隔的排除名稱")
    args = ap.parse_args()

    if args.list:
        print(f"{'name':<14s}{'size':<10s}description")
        print("-" * 70)
        for n, (_, desc, sz) in DATASETS.items():
            print(f"{n:<14s}{sz:<10s}{desc}")
        return

    names = list(DATASETS.keys())
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip() in DATASETS]
    if args.exclude:
        skip = {n.strip() for n in args.exclude.split(",")}
        names = [n for n in names if n not in skip]

    print(f"將下載 {len(names)} 個資料庫：{', '.join(names)}")
    print(f"輸出目錄：{BASE}")
    print()

    results = {}
    for n in names:
        fn, desc, sz = DATASETS[n]
        print(f"\n=== {n} ({desc}, {sz}) ===")
        t0 = time.time()
        ok = fn()
        results[n] = (ok, time.time() - t0)

    print("\n" + "=" * 60)
    print("總結：")
    for n, (ok, t) in results.items():
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {n:<14s} ({t/60:.1f} min)")


if __name__ == "__main__":
    main()
