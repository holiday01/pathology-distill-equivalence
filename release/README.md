# Checkpoint checksum manifest

`checksums_sha256.txt` is the SHA-256 manifest for the released student
checkpoints, distributed with the archived deposit (not in this repository).

It covers:

- the 36 headline-atlas checkpoints — 12 teachers × 3 student sizes, at
  `outputs/v4_full/<teacher>/<student>/best.pt`
- the 15 replicated-seed checkpoints backing the variance decomposition —
  5 teachers × 3 student sizes, `seed = 2025`, at
  `outputs/v4_seed2/<teacher>/<student>/best.pt`

Verify a downloaded bundle with:

```bash
sha256sum -c release/checksums_sha256.txt
```

`outputs/v4_full/hibou-l/vit-tiny_ep22_partial/best.pt` is intentionally
excluded: it is an earlier, superseded 22-epoch attempt made before the full
30-epoch run completed, and is not part of the reported 36-run atlas.
