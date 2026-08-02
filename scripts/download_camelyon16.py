#!/usr/bin/env python3
"""
從 AWS public bucket (s3://camelyon-dataset) 並行下載 CAMELYON16 WSI。

- 支援 resume（.part + Range）
- 跳過已存在、大小正確的檔案
- 8 個 worker
用法：
  python download_camelyon16.py --out ~/wsi_hl/data/camelyon16 \
      --normal 10 --tumor 10 --workers 8
"""

import argparse
import os
import sys
import time
import threading
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET = "https://camelyon-dataset.s3.amazonaws.com"
PREFIX = "CAMELYON16/images/"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_bucket(prefix=PREFIX):
    items, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        root = ET.fromstring(urllib.request.urlopen(url, timeout=30).read())
        for c in root.findall("s3:Contents", NS):
            items.append((c.find("s3:Key", NS).text, int(c.find("s3:Size", NS).text)))
        trunc = root.find("s3:IsTruncated", NS)
        if trunc is not None and trunc.text == "true":
            token = root.find("s3:NextContinuationToken", NS).text
        else:
            break
    return items


def fmt_bytes(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.2f}{u}"
        n /= 1024
    return f"{n:.2f}TB"


class Progress:
    def __init__(self, total_bytes, total_files):
        self.total_bytes = total_bytes
        self.total_files = total_files
        self.done_bytes = 0
        self.done_files = 0
        self.start = time.time()
        self.lock = threading.Lock()

    def add_bytes(self, n):
        with self.lock:
            self.done_bytes += n

    def finish_file(self):
        with self.lock:
            self.done_files += 1
        self.report()

    def report(self):
        elapsed = time.time() - self.start
        rate = self.done_bytes / elapsed if elapsed > 0 else 0
        remaining = (self.total_bytes - self.done_bytes) / rate if rate > 0 else float("inf")
        pct = 100 * self.done_bytes / self.total_bytes if self.total_bytes else 0
        print(
            f"  [{self.done_files}/{self.total_files}] "
            f"{fmt_bytes(self.done_bytes)}/{fmt_bytes(self.total_bytes)} "
            f"({pct:.1f}%)  {fmt_bytes(rate)}/s  ETA {remaining/60:.1f} min",
            flush=True,
        )


def download_one(key, size, out_dir: Path, progress: Progress):
    name = key.split("/")[-1]
    dst = out_dir / name
    part = out_dir / f"{name}.part"

    if dst.exists() and dst.stat().st_size == size:
        progress.add_bytes(size)
        progress.finish_file()
        return (name, "skip (已存在)")

    existing = part.stat().st_size if part.exists() else 0
    if existing >= size:
        part.rename(dst)
        progress.add_bytes(size)
        progress.finish_file()
        return (name, "skip (part 已完成)")

    url = f"{BUCKET}/{urllib.parse.quote(key)}"
    req = urllib.request.Request(url)
    if existing > 0:
        req.add_header("Range", f"bytes={existing}-")
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f:
            if existing > 0:
                progress.add_bytes(existing)
            while True:
                buf = r.read(1024 * 1024)
                if not buf:
                    break
                f.write(buf)
                progress.add_bytes(len(buf))
        part.rename(dst)
        progress.finish_file()
        return (name, "ok")
    except Exception as e:
        progress.finish_file()
        return (name, f"FAILED: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--normal", type=int, default=10)
    ap.add_argument("--tumor", type=int, default=10)
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("列出 bucket 檔案...", flush=True)
    items = list_bucket()
    tifs = [(k, s) for k, s in items if k.endswith(".tif")]
    name2size = {k.split("/")[-1]: (k, s) for k, s in tifs}

    existing = {p.name for p in out_dir.glob("*.tif")}

    def pick(prefix, n):
        pool = sorted(name for name in name2size if name.startswith(prefix + "_") and name not in existing)
        return pool[:n]

    chosen_names = pick("normal", args.normal) + pick("tumor", args.tumor) + pick("test", args.test)
    chosen = [(name2size[n][0], name2size[n][1]) for n in chosen_names]

    total = sum(s for _, s in chosen)
    print(f"目標檔案：{len(chosen)} 個，總大小 {fmt_bytes(total)}")
    for k, s in chosen:
        print(f"  {k.split('/')[-1]:30s} {fmt_bytes(s)}")

    if args.dry_run or not chosen:
        return

    print(f"\n開始下載（{args.workers} workers）→ {out_dir}", flush=True)
    progress = Progress(total, len(chosen))
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(download_one, k, s, out_dir, progress) for k, s in chosen]
        for f in as_completed(futures):
            results.append(f.result())

    print("\n下載完成摘要：")
    ok = sum(1 for _, st in results if st == "ok" or st.startswith("skip"))
    print(f"  成功: {ok}/{len(results)}")
    for name, st in results:
        if not (st == "ok" or st.startswith("skip")):
            print(f"  [失敗] {name}: {st}")


if __name__ == "__main__":
    main()
